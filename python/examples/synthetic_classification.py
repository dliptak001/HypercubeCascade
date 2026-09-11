#!/usr/bin/env python3
"""Synthetic multi-class fields → Cascade map → train → test.

Mirrors the spirit of the C++ ``cascade_synth`` demo using the public
``hypercube_cascade`` API only.

Onboarding demo (small dim, short train). Not a paper validator.

This file lives in the GitHub repo under ``python/examples/`` — it is not part
of ``pip install``. From a clone, repository root, after
``pip install hypercube-cascade`` (or ``pip install ./python``)::

    python python/examples/synthetic_classification.py

"""

from __future__ import annotations

import numpy as np

import hypercube_cascade as hc


def make_patterns(dim: int, n_per_class: int, n_classes: int, seed: int):
    rng = np.random.default_rng(seed)
    n = 1 << dim
    fields = []
    labels = []
    for c in range(n_classes):
        for rep in range(n_per_class):
            t = np.linspace(0, 2 * np.pi, n, dtype=np.float32)
            x = np.sin((c + 1) * t + 0.07 * rep).astype(np.float32)
            x += 0.12 * rng.standard_normal(n).astype(np.float32)
            x[n // 2 + (c % (n // 4))] += 1.2
            fields.append(x)
            labels.append(c)
    return np.stack(fields, axis=0), np.asarray(labels, dtype=np.int32)


def main() -> None:
    dim = 7
    n_classes = 6
    fields_tr, labels_tr = make_patterns(dim, 64, n_classes, seed=1)
    fields_te, labels_te = make_patterns(dim, 32, n_classes, seed=10_000)

    cas = hc.Cascade(
        dim=dim,
        T=50,
        interstage_scale=5.5,
        ic_seed=1,
        exciter_seed=3458567978345987,
        exciter_subcube_dim=5,
        exciter_input_scaling=1.0,
        exciter_weight_scaling=0.15,
        spectral_radius=0.9,
        history_depth=4,
        readout_num_outputs=n_classes,
        readout_task="classification",
        readout_epochs=100,
        readout_batch_size=32,
        readout_num_threads=1,
        readout_restore_best_epoch=False,
        collect_threads=0,
        verbose=True,
    )
    # collect → train (map API)
    cas.fit(fields_tr, labels_tr)

    train_acc = cas.accuracy_on_collected()
    test_acc = cas.accuracy(fields_te, labels_te)

    print(f"N={cas.N}  T={cas.T}  subcube_dim={cas.subcube_dim}  M={cas.M}")
    print(f"collected: {cas.num_collected}")
    print(f"train accuracy (collected set): {train_acc:.4f}")
    print(f"test accuracy (held-out fields): {test_acc:.4f}")
    print(f"live predict() shape: {cas.predict(fields_te[0]).shape}")


if __name__ == "__main__":
    main()
