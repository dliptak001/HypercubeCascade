"""Essential smoke tests for the hypercube_cascade wheel.

Kept lean so cibuildwheel stays short. Prove the compiled ``_core`` loads and
the map pipeline (collect → train → predict) produces sane results on the
target platform — not exhaustive façade coverage.
"""

from __future__ import annotations

import pickle

import numpy as np
import pytest

import hypercube_cascade
from hypercube_cascade import Cascade


def _make_patterns(dim: int, n_per_class: int, n_classes: int, seed: int = 0):
    """Simple multi-class length-N fields (tone-ish + noise)."""
    rng = np.random.default_rng(seed)
    n = 1 << dim
    fields = []
    labels = []
    for c in range(n_classes):
        for rep in range(n_per_class):
            t = np.linspace(0, 2 * np.pi, n, dtype=np.float32)
            x = np.sin((c + 1) * t + 0.1 * rep).astype(np.float32)
            x += 0.05 * rng.standard_normal(n).astype(np.float32)
            # Class-dependent peak in the high half
            x[n // 2 + c * 2] += 1.5
            fields.append(x)
            labels.append(c)
    return np.stack(fields, axis=0), np.asarray(labels, dtype=np.int32)


@pytest.fixture(scope="module")
def cls_data():
    return _make_patterns(dim=5, n_per_class=24, n_classes=3, seed=1)


@pytest.fixture(scope="module")
def trained_cls(cls_data):
    fields, labels = cls_data
    # Gains follow the C++ cascade_synth recipe: a weak default injection
    # (interstage 1.0 x reservoir input 0.02) crushes the features.
    cas = Cascade(
        dim=5,
        exciter_subcube_dim=4,
        exciter_seed=1,
        exciter_input_scaling=1.0,
        exciter_weight_scaling=0.15,
        reservoir_seed=1,
        spectral_radius=0.9,
        interstage_scale=5.5,
        ic_seed=2,
        history_depth=4,
        T=32,
        readout_num_outputs=3,
        readout_task="classification",
        readout_num_layers=1,
        readout_conv_channels=4,
        readout_use_pooling=False,
        readout_activation="none",
        readout_lr_max=0.003,
        readout_epochs=60,
        readout_batch_size=16,
        readout_num_threads=1,
        readout_restore_best_epoch=False,
        collect_threads=1,
    )
    cas.fit(fields, labels)
    return cas, fields, labels


# ── Construction ──

class TestVersion:
    def test_version_string(self):
        v = hypercube_cascade.__version__
        assert isinstance(v, str) and len(v) > 0
        assert v[0].isdigit()
        # Same string as the extension module (CMake baked from _version.py)
        from hypercube_cascade import _core
        assert _core.__version__ == v


class TestConstruction:
    @pytest.mark.parametrize("dim", [5, 7])
    def test_construct(self, dim):
        cas = Cascade(dim=dim, exciter_subcube_dim=4, history_depth=4, T=8)
        assert cas.dim == dim
        assert cas.N == 2**dim
        assert cas.num_collected == 0
        assert cas.T == 8
        assert cas.M == 4
        assert cas.subcube_dim == 4
        assert cas.walk_size == 16
        assert cas.feature_size == cas.N

    def test_invalid_dim(self):
        with pytest.raises(ValueError, match="dim must be"):
            Cascade(dim=4, exciter_subcube_dim=4)
        with pytest.raises(ValueError, match="dim must be"):
            Cascade(dim=13, exciter_subcube_dim=4)

    def test_invalid_subcube_dim(self):
        # C++ default subcube_dim=6 is illegal at dim 5; wrapper catches early.
        with pytest.raises(ValueError, match="exciter_subcube_dim"):
            Cascade(dim=5)
        with pytest.raises(ValueError, match="exciter_subcube_dim"):
            Cascade(dim=7, exciter_subcube_dim=8)

    def test_invalid_T(self):
        # No T=0 auto convention in Cascade — C++ throws, surfaces as ValueError.
        with pytest.raises(ValueError):
            Cascade(dim=6, T=0, history_depth=4)

    def test_invalid_gain(self):
        with pytest.raises(ValueError):
            Cascade(dim=6, interstage_scale=0.0, history_depth=4)
        with pytest.raises(ValueError):
            Cascade(dim=6, readout_scale=-1.0, history_depth=4)

    def test_defaults(self):
        cas = Cascade(dim=6, history_depth=4)
        assert cas.T == 100
        assert cas.interstage_scale == 1.0
        assert cas.readout_scale == 1.0
        assert cas.readout_task == "regression"
        assert cas.num_outputs == 1
        assert cas.subcube_dim == 6

    def test_repr(self):
        cas = Cascade(dim=5, exciter_subcube_dim=4, history_depth=4, T=8)
        r = repr(cas)
        assert "dim=5" in r
        assert "N=32" in r


# ── Classification pipeline ──

class TestClassification:
    def test_fit_train_acc(self, trained_cls):
        cas, _, _ = trained_cls
        acc = cas.accuracy_on_collected()
        assert acc > 0.85, f"train accuracy too low: {acc}"

    def test_predict_class_shape(self, trained_cls):
        cas, fields, labels = trained_cls
        pred = cas.predict_class(fields[0])
        assert isinstance(pred, int)
        assert 0 <= pred < cas.num_outputs
        logits = cas.predict(fields[0])
        assert logits.shape == (cas.num_outputs,)
        assert logits.dtype == np.float32
        assert int(np.argmax(logits)) == pred

    def test_heldout_sane(self, trained_cls):
        cas, _, _ = trained_cls
        # Fresh draws with different seed — should still beat chance
        fields_te, labels_te = _make_patterns(5, 16, 3, seed=99)
        acc = cas.accuracy(fields_te, labels_te)
        assert acc > 0.5, f"test accuracy too low: {acc}"
        # Bulk accuracy agrees with per-sample predict_class
        correct = sum(
            cas.predict_class(fields_te[i]) == int(labels_te[i])
            for i in range(len(labels_te))
        )
        assert acc == pytest.approx(correct / len(labels_te))


# ── Regression ──

class TestRegression:
    def test_r2_on_collected(self):
        dim = 5
        n = 1 << dim
        rng = np.random.default_rng(3)
        fields = rng.standard_normal((80, n), dtype=np.float32)
        # Strong scalar signal in the field so a short train can move R²
        targets = (2.0 * fields[:, 0:1] + 0.05 * rng.standard_normal((80, 1))).astype(
            np.float32
        )
        cas = Cascade(
            dim=dim,
            exciter_subcube_dim=4,
            exciter_input_scaling=1.0,
            exciter_weight_scaling=0.15,
            history_depth=4,
            T=8,
            reservoir_input_scaling=0.5,
            readout_num_outputs=1,
            readout_task="regression",
            readout_epochs=80,
            readout_num_threads=1,
            collect_threads=1,
            readout_restore_best_epoch=False,
        )
        cas.fit(fields, targets)
        r2 = cas.r2_on_collected()
        assert np.isfinite(r2), f"R² not finite: {r2}"
        # Smoke: should beat "always predict mean" by a bit on this easy signal
        assert r2 > 0.0, f"R² too low: {r2}"
        y = cas.predict(fields[0])
        assert y.shape == (1,)
        assert y.dtype == np.float32
        # Held-out R² is finite on fresh draws
        fields_te = rng.standard_normal((20, n), dtype=np.float32)
        targets_te = (2.0 * fields_te[:, 0:1]).astype(np.float32)
        assert np.isfinite(cas.r2(fields_te, targets_te))


# ── Map helpers ──

class TestMap:
    def test_run_last_probes(self):
        cas = Cascade(dim=5, exciter_subcube_dim=4, history_depth=4, T=8)
        x = np.zeros(cas.N, dtype=np.float32)
        x[0] = 1.0
        cas.run(x)
        feat = cas.last_features()
        assert feat.shape == (cas.feature_size,)
        assert feat.dtype == np.float32
        for probe in (cas.last_exciter, cas.last_interstage, cas.last_reservoir):
            arr = probe()
            assert arr.shape == (cas.N,)
            assert arr.dtype == np.float32
        # interstage = exciter * interstage_scale (1.0 here)
        np.testing.assert_allclose(
            cas.last_interstage(), cas.last_exciter(), atol=1e-6)
        # features = reservoir * readout_scale (1.0 here)
        np.testing.assert_allclose(
            cas.last_features(), cas.last_reservoir(), atol=1e-6)

    def test_deterministic_map(self):
        cas = Cascade(dim=5, exciter_subcube_dim=4, history_depth=4, T=8)
        x = np.linspace(-1, 1, cas.N, dtype=np.float32)
        cas.run(x)
        a = cas.last_features().copy()
        cas.run(x)
        b = cas.last_features().copy()
        np.testing.assert_array_equal(a, b)

    def test_field_size_check(self):
        cas = Cascade(dim=5, exciter_subcube_dim=4, history_depth=4, T=8)
        with pytest.raises(Exception, match="must equal N"):
            cas.run(np.zeros(16, dtype=np.float32))


# ── Serial collect ──

class TestSerialCollect:
    def test_collect_appends(self, cls_data):
        fields, labels = cls_data
        cas = Cascade(
            dim=5,
            exciter_subcube_dim=4,
            history_depth=4,
            T=8,
            readout_num_outputs=3,
            readout_task="classification",
            collect_threads=1,
        )
        assert cas.num_collected == 0
        cas.collect(fields[0], int(labels[0]))
        cas.collect(fields[1], int(labels[1]))
        assert cas.num_collected == 2
        # Serial collect leaves this sample's features in last_features
        assert cas.last_features().shape == (cas.N,)
        cas.clear_collected()
        assert cas.num_collected == 0

    def test_collect_rejects_bool_label(self, cls_data):
        fields, _ = cls_data
        cas = Cascade(
            dim=5,
            exciter_subcube_dim=4,
            history_depth=4,
            T=8,
            readout_num_outputs=3,
            readout_task="classification",
        )
        with pytest.raises(TypeError, match="integer"):
            cas.collect(fields[0], True)

    def test_label_out_of_range(self, cls_data):
        fields, _ = cls_data
        cas = Cascade(
            dim=5,
            exciter_subcube_dim=4,
            history_depth=4,
            T=8,
            readout_num_outputs=3,
            readout_task="classification",
        )
        with pytest.raises(ValueError):
            cas.collect(fields[0], 3)


# ── Persistence ──

class TestPersistence:
    def test_pickle_roundtrip(self, trained_cls):
        cas, fields, labels = trained_cls
        acc_before = cas.accuracy_on_collected()
        loaded = pickle.loads(pickle.dumps(cas))
        assert loaded.num_collected == 0
        assert loaded.dim == cas.dim
        assert loaded.subcube_dim == cas.subcube_dim
        # Same weights → same predictions
        for i in range(8):
            assert loaded.predict_class(fields[i]) == cas.predict_class(fields[i])
        # Retrain not required for infer
        assert acc_before > 0.85

    def test_save_load(self, trained_cls, tmp_path):
        cas, fields, _ = trained_cls
        path = tmp_path / "model.pkl"
        cas.save(path)
        loaded = Cascade.load(path)
        assert loaded.predict_class(fields[0]) == cas.predict_class(fields[0])

    def test_hcnn_model_roundtrip(self, trained_cls, tmp_path):
        cas, fields, _ = trained_cls
        stem = tmp_path / "export" / "stem"
        (tmp_path / "export").mkdir()
        cas.save_readout_hcnn_model(stem)
        assert (tmp_path / "export" / "stem.hcnw").is_file()
        assert (tmp_path / "export" / "stem.arch.json").is_file()
        # Fresh instance, same architecture knobs → identical predictions
        fresh = Cascade(
            dim=5,
            exciter_subcube_dim=4,
            exciter_seed=1,
            exciter_input_scaling=1.0,
            exciter_weight_scaling=0.15,
            reservoir_seed=1,
            spectral_radius=0.9,
            interstage_scale=5.5,
            ic_seed=2,
            history_depth=4,
            T=32,
            readout_num_outputs=3,
            readout_task="classification",
            readout_num_layers=1,
            readout_conv_channels=4,
            readout_use_pooling=False,
            readout_activation="none",
            readout_num_threads=1,
        )
        fresh.load_readout_hcnn_model(stem)
        for i in range(8):
            assert fresh.predict_class(fields[i]) == cas.predict_class(fields[i])
        summary = cas.readout_arch_summary()
        assert isinstance(summary, str) and "weight_count" in summary


# ── Surface ──

class TestSurface:
    EXPECTED = [
        "run",
        "last_features",
        "last_exciter",
        "last_interstage",
        "last_reservoir",
        "clear_collected",
        "collect",
        "collect_batch",
        "fit",
        "train",
        "predict",
        "predict_class",
        "accuracy_on_collected",
        "r2_on_collected",
        "accuracy",
        "r2",
        "save",
        "load",
        "save_readout_hcnn_model",
        "load_readout_hcnn_model",
        "readout_arch_summary",
    ]

    @pytest.mark.parametrize("name", EXPECTED)
    def test_method_present(self, name):
        assert callable(getattr(Cascade, name, None)), f"Cascade.{name} missing"
