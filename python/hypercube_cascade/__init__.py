"""HypercubeCascade: frozen etalon transit + frozen reservoir orbit + HypercubeCNN.

Static length-N fields (no intrinsic time): one etalon transit, then a short
reservoir orbit per sample, then a HypercubeCNN readout trained on the
end-state features only.

Quick start::

    import numpy as np
    import hypercube_cascade as hc

    # fields: (num_samples, N) float32, labels: (num_samples,) int
    cas = hc.Cascade(dim=7, exciter_subcube_dim=5, readout_num_outputs=6,
                     readout_task="classification", readout_epochs=80)
    cas.fit(fields_train, labels_train)
    pred = cas.predict_class(fields_test[0])
"""

from __future__ import annotations

import pathlib
import pickle

import numpy as np

from ._core import _Cascade
from ._version import __version__

__all__ = ["Cascade", "__version__"]

# Valid hypercube dimensions (matches C++ Cascade constructor [5, 12] check).
_DIM_MIN = 5
_DIM_MAX = 12


def _to_float32(arr):
    """Ensure array is C-contiguous float32."""
    return np.ascontiguousarray(arr, dtype=np.float32)


def _to_int32(arr):
    """Ensure array is C-contiguous int32 (labels)."""
    return np.ascontiguousarray(arr, dtype=np.int32)


class Cascade:
    """HypercubeCascade product: field → etalon transit → reservoir orbit → HCNN.

    One cube dimension serves all three stages: N = 2^dim is the field length,
    the transit output length, and the feature length. Each sample is a full
    length-N field (you pack domain data on the host). A **map** runs one
    frozen etalon transit, scales it by ``interstage_scale``, reloads a frozen
    IC, drives the reservoir for T xor-re-addressed passes, and hands the end
    state times ``readout_scale`` to the readout.

    Typical lifecycle: :meth:`collect_batch` → :meth:`train` →
    :meth:`predict` / :meth:`predict_class`, or the one-shot :meth:`fit`.

    Parameters
    ----------
    dim : int
        Hypercube dimension (5–12). N = 2^dim field length, all stages.
    T : int
        Reservoir drive-pass count per sample. Must be >= 1 (no auto value).
        Default: 100.
    interstage_scale : float
        Gain on the transit output before the orbit. Finite, > 0. Default: 1.0.
    readout_scale : float
        Gain on the reservoir end state before the readout. Finite, > 0.
        Default: 1.0.
    ic_seed : int
        Seed for the frozen episode initial condition (separate from the
        weight seeds). Default: 1.
    collect_threads : int
        Workers for bulk collect / scoring: 0 = auto, 1 = serial, K = K workers.
    exciter_seed : int
        Exciter weight-init seed. Default matches C++ ``ExciterConfig``.
    exciter_input_scaling : float
        Scalar applied once to the field before the transit. Default: 0.02.
    exciter_weight_scaling : float
        Exciter neighbor weights are U(-1, 1) x this. Default: 0.02.
    exciter_subcube_dim : int
        Etalon face size; the walk covers 2^subcube_dim vertices. Valid
        [1, dim] — **the default 6 throws below dim 6**; set it <= dim.
    reservoir_seed : int
        Reservoir weight-init seed. Default matches C++ ``ReservoirConfig``.
    spectral_radius : float
        Target spectral radius for recurrent weights. Default: 0.999.
    reservoir_input_scaling : float
        Reservoir input drive coefficient. Default: 0.02.
    leak_rate : float
        Leaky integrator; 1.0 = full replacement. Default: 1.0.
    history_depth : int
        Delay-line depth M in [1, 64]. Default: 16.
    bias_scaling : float
        Per-neuron bias scale after tanh; 0 disables. Default: 0.003.
    verbose : bool
        Print the reservoir construction banner. Default: False.
    readout_num_outputs : int
        Classes (classification) or regression width. Default: 1.
    readout_task : str
        ``"regression"`` (default) or ``"classification"``.
    readout_num_layers : int
        Conv(+Pool) pairs. Default: 1. 0 = auto min(dim-2, 2).
    readout_conv_channels : int
        Base channel count. Default: 16.
    readout_epochs : int
        Batch train epochs. Default: 200.
    readout_batch_size : int
        Mini-batch size. Default: 32.
    readout_lr_max : float
        Cosine peak LR. Default: 0.0015.
    readout_lr_min_frac : float
        Floor as fraction of lr_max. Default: 0.01.
    readout_lr_decay_epochs : int
        Cosine horizon; 0 = use epochs. Default: 0.
    readout_weight_decay : float
        L2 on CNN weights. Default: 0.0.
    readout_momentum : float
        SGD momentum (ignored by Adam). Default: 0.9.
    readout_activation : str
        ``"tanh"`` (default), ``"relu"``, ``"leaky_relu"``, or ``"none"``.
    readout_seed : int
        CNN weight-init seed. Default: 42.
    readout_num_threads : int
        HCNN pool: 0 = auto, 1 = single-threaded. Default: 0.
    readout_restore_best_epoch : bool
        Restore best-epoch weights after batch train. Default: True.
    readout_best_epoch_holdout_frac : float
        Tail hold-out for best-epoch scoring. Default: 0.0.
    readout_use_pooling : bool
        Antipodal pool after each conv. Default: True.

    Notes
    -----
    This class is **not thread-safe** for concurrent public calls from multiple
    host threads. Bulk collect / scoring parallelism is internal.

    Examples
    --------
    >>> import numpy as np
    >>> import hypercube_cascade as hc
    >>> rng = np.random.default_rng(0)
    >>> N = 32  # dim=5
    >>> fields = rng.standard_normal((64, N), dtype=np.float32)
    >>> labels = rng.integers(0, 3, size=64)
    >>> cas = hc.Cascade(dim=5, exciter_subcube_dim=4, readout_num_outputs=3,
    ...                  readout_task="classification", readout_epochs=40,
    ...                  history_depth=4, T=16)
    >>> cas.fit(fields, labels)
    Cascade(dim=5, N=32, ...)
    """

    def __init__(
        self,
        dim: int,
        *,
        T: int = 100,
        interstage_scale: float = 1.0,
        readout_scale: float = 1.0,
        ic_seed: int = 1,
        collect_threads: int = 0,
        exciter_seed: int = 7934791766227647176,
        exciter_input_scaling: float = 0.02,
        exciter_weight_scaling: float = 0.02,
        exciter_subcube_dim: int = 6,
        reservoir_seed: int = 7934791766227647176,
        spectral_radius: float = 0.999,
        reservoir_input_scaling: float = 0.02,
        leak_rate: float = 1.0,
        history_depth: int = 16,
        bias_scaling: float = 0.003,
        verbose: bool = False,
        readout_num_outputs: int = 1,
        readout_task: str = "regression",
        readout_num_layers: int = 1,
        readout_conv_channels: int = 16,
        readout_epochs: int = 200,
        readout_batch_size: int = 32,
        readout_lr_max: float = 0.0015,
        readout_lr_min_frac: float = 0.01,
        readout_lr_decay_epochs: int = 0,
        readout_weight_decay: float = 0.0,
        readout_momentum: float = 0.9,
        readout_activation: str = "tanh",
        readout_seed: int = 42,
        readout_num_threads: int = 0,
        readout_restore_best_epoch: bool = True,
        readout_best_epoch_holdout_frac: float = 0.0,
        readout_use_pooling: bool = True,
    ):
        if not isinstance(dim, int) or not (_DIM_MIN <= dim <= _DIM_MAX):
            raise ValueError(
                f"dim must be an integer in [{_DIM_MIN}, {_DIM_MAX}], got {dim!r}"
            )
        if not isinstance(exciter_subcube_dim, int) or not (
            1 <= exciter_subcube_dim <= dim
        ):
            raise ValueError(
                f"exciter_subcube_dim must be an integer in [1, {dim}] "
                f"(got {exciter_subcube_dim!r}; the C++ default 6 is only "
                f"legal when dim >= 6)"
            )
        if readout_task not in ("regression", "classification"):
            raise ValueError(
                f"readout_task must be 'regression' or 'classification', "
                f"got {readout_task!r}"
            )
        if readout_activation not in ("tanh", "relu", "leaky_relu", "none"):
            raise ValueError(
                "readout_activation must be one of "
                "'tanh', 'relu', 'leaky_relu', 'none' "
                f"(got {readout_activation!r})"
            )
        self._verbose = verbose
        self._ctor = {
            "dim": dim,
            "T": T,
            "interstage_scale": interstage_scale,
            "readout_scale": readout_scale,
            "ic_seed": ic_seed,
            "collect_threads": collect_threads,
            "exciter_seed": exciter_seed,
            "exciter_input_scaling": exciter_input_scaling,
            "exciter_weight_scaling": exciter_weight_scaling,
            "exciter_subcube_dim": exciter_subcube_dim,
            "reservoir_seed": reservoir_seed,
            "spectral_radius": spectral_radius,
            "reservoir_input_scaling": reservoir_input_scaling,
            "leak_rate": leak_rate,
            "history_depth": history_depth,
            "bias_scaling": bias_scaling,
            "verbose": verbose,
            "readout_num_outputs": readout_num_outputs,
            "readout_task": readout_task,
            "readout_num_layers": readout_num_layers,
            "readout_conv_channels": readout_conv_channels,
            "readout_epochs": readout_epochs,
            "readout_batch_size": readout_batch_size,
            "readout_lr_max": readout_lr_max,
            "readout_lr_min_frac": readout_lr_min_frac,
            "readout_lr_decay_epochs": readout_lr_decay_epochs,
            "readout_weight_decay": readout_weight_decay,
            "readout_momentum": readout_momentum,
            "readout_activation": readout_activation,
            "readout_seed": readout_seed,
            "readout_num_threads": readout_num_threads,
            "readout_restore_best_epoch": readout_restore_best_epoch,
            "readout_best_epoch_holdout_frac": readout_best_epoch_holdout_frac,
            "readout_use_pooling": readout_use_pooling,
        }
        self._impl = _Cascade(**self._ctor)

    # ── Map (no training) ──

    def run(self, x: np.ndarray) -> None:
        """Map one field (transit → orbit → features), no training-set append.

        Updates :meth:`last_features` and the per-stage probes
        :meth:`last_exciter` / :meth:`last_interstage` / :meth:`last_reservoir`.

        Parameters
        ----------
        x : ndarray
            Length-N field (or shape that ravel-flattens to N). Converted to
            float32.
        """
        self._impl.run(_to_float32(np.ravel(x)))

    def last_features(self) -> np.ndarray:
        """Features (length N) from the most recent completed map.

        Updated by every map on this instance, including bulk calls
        (last row).
        """
        return self._impl.last_features()

    def last_exciter(self) -> np.ndarray:
        """Etalon transit output from the last **serial** map (length N).

        Not updated by bulk :meth:`collect_batch` / :meth:`accuracy` /
        :meth:`r2`.
        """
        return self._impl.last_exciter()

    def last_interstage(self) -> np.ndarray:
        """Transit output × interstage_scale (the reservoir drive) from the
        last **serial** map."""
        return self._impl.last_interstage()

    def last_reservoir(self) -> np.ndarray:
        """Reservoir end state (before readout_scale) from the last **serial**
        map."""
        return self._impl.last_reservoir()

    def clear_collected(self) -> None:
        """Drop all samples collected for batch training."""
        self._impl.clear_collected()

    # ── Collect ──

    def collect(
        self,
        x: np.ndarray,
        target,
    ) -> None:
        """Serial collect one sample (classification label or regression target).

        Parameters
        ----------
        x : ndarray
            Length-N field.
        target : int or ndarray
            Class index (classification) or length-``num_outputs`` floats
            (regression).
        """
        field = _to_float32(np.ravel(x))
        if self._ctor["readout_task"] == "classification":
            if isinstance(target, (bool, np.bool_)):
                raise TypeError("class label must be an integer")
            if isinstance(target, (int, np.integer)):
                label = int(target)
            else:
                arr = np.asarray(target).ravel()
                if arr.size != 1:
                    raise ValueError("classification target must be a single class index")
                label = int(arr[0])
            self._impl.collect_class(field, label)
        else:
            t = _to_float32(np.ravel(target))
            self._impl.collect_reg(field, t)

    def _check_bulk_fields(self, fields: np.ndarray):
        fields = _to_float32(fields)
        n = self.N
        if fields.ndim == 2:
            if fields.shape[1] != n:
                raise ValueError(
                    f"fields.shape[1] ({fields.shape[1]}) must equal N ({n})"
                )
            count = int(fields.shape[0])
            flat = np.ascontiguousarray(fields.reshape(-1))
        elif fields.ndim == 1:
            if fields.size % n != 0:
                raise ValueError(
                    f"flat fields size ({fields.size}) must be a multiple of N ({n})"
                )
            count = int(fields.size // n)
            flat = fields
        else:
            raise ValueError(
                f"fields must be 1D or 2D, got ndim={fields.ndim}"
            )
        return flat, count

    def _check_bulk_targets(self, targets: np.ndarray, count: int):
        if self._ctor["readout_task"] == "classification":
            labels = _to_int32(np.ravel(targets))
            if labels.size != count:
                raise ValueError(
                    f"labels length ({labels.size}) must equal sample count ({count})"
                )
            return labels
        t = _to_float32(targets)
        k = self.num_outputs
        if t.ndim == 2:
            if t.shape[0] != count or t.shape[1] != k:
                raise ValueError(
                    f"targets shape must be ({count}, {k}), got {t.shape}"
                )
            t = np.ascontiguousarray(t.reshape(-1))
        elif t.ndim == 1:
            if t.size != count * k:
                raise ValueError(
                    f"flat targets size ({t.size}) must equal "
                    f"count * num_outputs ({count * k})"
                )
        else:
            raise ValueError(f"targets must be 1D or 2D, got ndim={t.ndim}")
        return t

    def collect_batch(
        self,
        fields: np.ndarray,
        targets: np.ndarray,
    ) -> None:
        """Bulk parallel collect (appends to the training set).

        Parameters
        ----------
        fields : ndarray
            Shape ``(count, N)`` preferred, or flat length ``count * N``.
        targets : ndarray
            Classification: shape ``(count,)`` integer labels.
            Regression: shape ``(count, num_outputs)`` or flat ``count * num_outputs``.
        """
        flat, count = self._check_bulk_fields(fields)
        t = self._check_bulk_targets(targets, count)
        if self._ctor["readout_task"] == "classification":
            self._impl.collect_batch_class(flat, t)
        else:
            self._impl.collect_batch_reg(flat, t)

    def fit(
        self,
        fields: np.ndarray,
        targets: np.ndarray,
        *,
        clear: bool = True,
    ) -> "Cascade":
        """Collect then train the readout (one-shot).

        Parameters
        ----------
        fields : ndarray
            Shape ``(count, N)`` length-N fields (host packing is your problem).
        targets : ndarray
            Labels or regression targets (see :meth:`collect_batch`).
        clear : bool
            If True (default), :meth:`clear_collected` first.

        Returns
        -------
        Cascade
            Self, for method chaining.
        """
        if clear:
            self.clear_collected()
        self.collect_batch(fields, targets)
        self.train()
        return self

    def train(self) -> None:
        """Batch-train the HCNN on all collected samples.

        Does not clear the collected set — call again to continue training
        from the current weights, or :meth:`clear_collected` first to start
        over.
        """
        self._impl.train()

    # ── Inference ──

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Fresh map + readout forward.

        Parameters
        ----------
        x : ndarray
            Length-N field.

        Returns
        -------
        ndarray
            Shape ``(num_outputs,)`` float32 logits / regression values.
        """
        return self._impl.predict(_to_float32(np.ravel(x)))

    def predict_class(self, x: np.ndarray) -> int:
        """Fresh map + argmax class (classification task only)."""
        return int(self._impl.predict_class(_to_float32(np.ravel(x))))

    def accuracy_on_collected(self) -> float:
        """Accuracy on the **collected training set** — not a test-set metric."""
        return float(self._impl.accuracy_on_collected())

    def r2_on_collected(self) -> float:
        """R² on the **collected training set** — not a test-set metric."""
        return float(self._impl.r2_on_collected())

    def accuracy(self, fields: np.ndarray, labels: np.ndarray) -> float:
        """Fresh map + accuracy on a caller-owned (held-out) set.

        Maps every field through the cascade (bulk, parallel) and scores the
        readout. Classification task only.

        Parameters
        ----------
        fields : ndarray
            Shape ``(count, N)`` or flat ``count * N``.
        labels : ndarray
            Shape ``(count,)`` integer class indices.
        """
        flat, count = self._check_bulk_fields(fields)
        lab = _to_int32(np.ravel(labels))
        if lab.size != count:
            raise ValueError(
                f"labels length ({lab.size}) must equal sample count ({count})"
            )
        return float(self._impl.accuracy(flat, lab))

    def r2(self, fields: np.ndarray, targets: np.ndarray) -> float:
        """Fresh map + R² on a caller-owned (held-out) set.

        Regression task only.

        Parameters
        ----------
        fields : ndarray
            Shape ``(count, N)`` or flat ``count * N``.
        targets : ndarray
            Shape ``(count, num_outputs)`` or flat ``count * num_outputs``.
        """
        flat, count = self._check_bulk_fields(fields)
        t = self._check_bulk_targets(targets, count)
        return float(self._impl.r2(flat, t))

    # ── Properties ──

    @property
    def dim(self) -> int:
        """Hypercube dimension (all three stages)."""
        return int(self._impl.dim)

    @property
    def N(self) -> int:
        """Field / feature length: 2^dim."""
        return int(self._impl.N)

    @property
    def T(self) -> int:
        """Reservoir drive-pass count per sample."""
        return int(self._impl.T)

    @property
    def M(self) -> int:
        """Delay-line depth (history_depth)."""
        return int(self._impl.history_depth)

    @property
    def feature_size(self) -> int:
        """Floats per collected sample / last_features — always N."""
        return int(self._impl.N)

    @property
    def num_collected(self) -> int:
        """Number of samples in the batch training set."""
        return int(self._impl.num_collected)

    @property
    def num_outputs(self) -> int:
        """Readout output width."""
        return int(self._impl.num_outputs)

    @property
    def collect_threads(self) -> int:
        """Configured collect-thread preference (0 = auto)."""
        return int(self._impl.collect_threads)

    @property
    def interstage_scale(self) -> float:
        """Gain between transit output and reservoir drive."""
        return float(self._impl.interstage_scale)

    @property
    def readout_scale(self) -> float:
        """Gain between reservoir end state and readout."""
        return float(self._impl.readout_scale)

    @property
    def ic_seed(self) -> int:
        """Frozen episode IC seed."""
        return int(self._impl.ic_seed)

    @property
    def exciter_seed(self) -> int:
        """Exciter weight seed."""
        return int(self._impl.exciter_seed)

    @property
    def exciter_input_scaling(self) -> float:
        return float(self._impl.exciter_input_scaling)

    @property
    def exciter_weight_scaling(self) -> float:
        return float(self._impl.exciter_weight_scaling)

    @property
    def subcube_dim(self) -> int:
        """Etalon face dimension."""
        return int(self._impl.subcube_dim)

    @property
    def walk_size(self) -> int:
        """Vertices on one etalon out-and-back: 2^subcube_dim."""
        return int(self._impl.walk_size)

    @property
    def reservoir_seed(self) -> int:
        """Reservoir weight seed."""
        return int(self._impl.reservoir_seed)

    @property
    def spectral_radius(self) -> float:
        """Target spectral radius (config)."""
        return float(self._impl.spectral_radius)

    @property
    def realized_spectral_radius(self) -> float:
        """Post-rescale realized spectral-radius estimate."""
        return float(self._impl.realized_spectral_radius)

    @property
    def reservoir_input_scaling(self) -> float:
        return float(self._impl.reservoir_input_scaling)

    @property
    def leak_rate(self) -> float:
        return float(self._impl.leak_rate)

    @property
    def history_depth(self) -> int:
        return int(self._impl.history_depth)

    @property
    def bias_scaling(self) -> float:
        return float(self._impl.bias_scaling)

    @property
    def verbose(self) -> bool:
        return bool(self._verbose)

    @property
    def readout_task(self) -> str:
        return str(self._impl.readout_task)

    @property
    def readout_best_epoch(self) -> int:
        """1-based best epoch after restore_best_epoch train, else 0."""
        return int(self._impl.readout_best_epoch)

    def __repr__(self) -> str:
        return (
            f"Cascade(dim={self.dim}, N={self.N}, T={self.T}, "
            f"subcube_dim={self.subcube_dim}, "
            f"collected={self.num_collected}, outputs={self.num_outputs}, "
            f"task={self.readout_task})"
        )

    # ── Persistence ──

    _PERSISTENCE_VERSION = 1

    def __getstate__(self) -> dict:
        """Serialize constructor config + trained readout weights.

        Collected samples are **not** saved.
        """
        return {
            "_version": self._PERSISTENCE_VERSION,
            "ctor": dict(self._ctor),
            "readout_state": self._impl._get_readout_state(),
        }

    def __setstate__(self, state: dict) -> None:
        version = state.get("_version", 0)
        if version > self._PERSISTENCE_VERSION:
            raise ValueError(
                f"Model was saved with persistence version {version}, "
                f"but this version only supports up to "
                f"{self._PERSISTENCE_VERSION}. Upgrade hypercube-cascade."
            )
        ctor = dict(state["ctor"])
        self.__init__(**ctor)
        self._impl._set_readout_state(state["readout_state"])

    def save(self, path) -> None:
        """Save config + trained readout to a pickle file.

        Collected samples are not saved. Prefer
        :meth:`save_readout_hcnn_model` for portable HCNW + arch JSON.
        """
        with open(pathlib.Path(path), "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path) -> "Cascade":
        """Load a model saved by :meth:`save`.

        .. warning::

            Uses ``pickle.load``. Never load untrusted files.
        """
        with open(pathlib.Path(path), "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected Cascade, got {type(obj).__name__}")
        return obj

    def save_readout_hcnn_model(self, path_stem) -> None:
        """Export the HCNN readout as portable ``stem.hcnw`` + ``stem.arch.json``."""
        self._impl.save_readout_hcnn_model(str(path_stem))

    def load_readout_hcnn_model(self, path_stem, *, mode: str = "eval") -> None:
        """Load ``stem.hcnw`` (+ arch sidecar) into this instance's readout.

        Parameters
        ----------
        path_stem : str or Path
            Path without extension.
        mode : str
            ``"eval"`` (default) or ``"resume_train"``.
        """
        self._impl.load_readout_hcnn_model(str(path_stem), mode)

    def readout_arch_summary(self) -> str:
        """Human-readable HCNN readout architecture and parameter counts."""
        return self._impl.readout_arch_summary()
