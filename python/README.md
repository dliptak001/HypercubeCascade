# Hypercube Cascade

[![Build wheels](https://github.com/dliptak001/HypercubeCascade/actions/workflows/wheels.yml/badge.svg)](https://github.com/dliptak001/HypercubeCascade/actions/workflows/wheels.yml)
[![PyPI](https://img.shields.io/pypi/v/hypercube-cascade)](https://pypi.org/project/hypercube-cascade/)
[![Python](https://img.shields.io/pypi/pyversions/hypercube-cascade)](https://pypi.org/project/hypercube-cascade/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/dliptak001/HypercubeCascade/blob/main/LICENSE)
[![C++23](https://img.shields.io/badge/C%2B%2B-23-blue.svg)](https://en.cppreference.com/w/cpp/23)

This package is the **Python** surface for HypercubeCascade
(`import hypercube_cascade`).
Full API reference: **[docs/Python_SDK.md](https://github.com/dliptak001/HypercubeCascade/blob/main/docs/Python_SDK.md)**.
C++ integration guide: **[docs/CPP_SDK.md](https://github.com/dliptak001/HypercubeCascade/blob/main/docs/CPP_SDK.md)**.
Project home: **[github.com/dliptak001/HypercubeCascade](https://github.com/dliptak001/HypercubeCascade)**.

HypercubeCascade processes spatial data of the kind presented to a CNN.
It is built from four core classes.

The **Cascade** class wraps the other three and manages training and
prediction.

The other three form a pipeline: etalon → reservoir → readout.

The **Exciter** class is a preprocessing stage that consumes input
patterns, mixes them nonlinearly, and returns a field with the same
dimensions as the input.

The **Reservoir** class is a second preprocessing stage that consumes
that field, drives a short synthetic orbit on a frozen hypercube
reservoir, and returns a field with the same dimensions.

The **Readout** class is a small HypercubeCNN that classifies or
regresses that field.

This is reservoir computing, but not only reservoir computing.

The point of this experiment is to see if two preprocessing stages in
front of HypercubeCNN outperform HypercubeCNN by itself, and
outperform either stage alone. HypercubeEtalon is the etalon alone.
HypercubeWTF is the reservoir alone. Cascade runs them in series.
The aim is a hypercube preprocessor effective enough that the readout
can be a single layer with a single convolutional channel and no
pooling. Then training is fast, the memory footprint is small, and
little to no architectural engineering is required for the CNN.

---

<p align="center">
  <strong>HypercubeAI ecosystem</strong><br/>
</p>

<p align="center">
  <a href="https://github.com/dliptak001/HypercubeCascade"><strong>HypercubeCascade</strong></a>
  &nbsp;·&nbsp;
  <a href="https://github.com/dliptak001/HypercubeCNN"><strong>HypercubeCNN</strong></a>
  &nbsp;·&nbsp;
  <a href="https://github.com/dliptak001/HypercubeESN"><strong>HypercubeESN</strong></a>
  &nbsp;·&nbsp;
  <a href="https://github.com/dliptak001/HypercubeEtalon"><strong>HypercubeEtalon</strong></a>
  &nbsp;·&nbsp;
  <a href="https://github.com/dliptak001/HypercubeHopfield"><strong>HypercubeHopfield</strong></a>
  &nbsp;·&nbsp;
  <a href="https://github.com/dliptak001/HypercubeLCN"><strong>HypercubeLCN</strong></a>
  &nbsp;·&nbsp;
  <a href="https://github.com/dliptak001/HypercubeWorldModel"><strong>HypercubeWorldModel</strong></a>
  &nbsp;·&nbsp;
  <a href="https://github.com/dliptak001/HypercubeWTF"><strong>HypercubeWTF</strong></a>
</p>

<p align="center">
  📄 Foundational paper:
  <a href="https://github.com/dliptak001/HypercubeCascade/blob/main/docs/Boolean_hypercubes_as_a_neural_substrate.pdf"><em>Boolean Hypercubes as a Neural Substrate</em></a>
  (D.&nbsp;C.&nbsp;Liptak, 2026)
</p>

HypercubeCascade is an experiment in the **HypercubeAI** project — our quest to
systematically re-implement classical neural architectures on a Boolean
hypercube topology instead of Euclidean grids or random graphs. The central
thesis is “topology-native intelligence”: the hypercube’s algebraic structure
(vertex-transitive symmetry, Hamming geometry, bitwise addressing) can serve
as a first-class computational substrate.

- **A topology you don’t store** — the graph is specified: connectivity is
  implicit in the vertex indices; with a seed and a few config scalars the whole
  reservoir reconstructs mathematically.
- **Perfect homogeneity** — every vertex has the same degree and the same local
  world, so local dynamics mean the same thing everywhere — no structural
  favorites baked in by a random graph.
- **Cheap navigation** — each neighbor is a few bit operations on the vertex
  index, not a pointer chase through a stored edge list, so walks stay
  arithmetic and cache-friendly.
- **Topology-native pairing** — the readout consumes the reservoir’s output with
  zero geometric distortion, and the learned kernels exploit the same locality
  that generated the dynamics. The data never leaves the hypercube it was born
  on.

Each product in the family is a different architecture on that same foundation.

---

## The Cascade

HypercubeEtalon and HypercubeWTF are two examples of how
solutions can be built on that substrate. Cascade is both of them, in
series, on one cube.

There is one cube dimension. The Exciter, the Reservoir, and the
Readout all use it.

An etalon, here, is a vertex and its antipode treated as a reflective
cavity. The Exciter walks every such cavity and writes one output
sample per start. That walk is the etalon transit. The write-up is
[HypercubeEtalon](https://github.com/dliptak001/HypercubeEtalon).

The Reservoir is the HypercubeWTF encoder: frozen recurrent weights,
a delay line, and a short synthetic orbit. Geometry stays put; the
registration of the field moves. The write-up is
[HypercubeWTF](https://github.com/dliptak001/HypercubeWTF).

The cascade itself goes something like this.

    Copy the input field. Never write the caller's buffer.

    Run one etalon transit. The cube is the same size it started as.

    Multiply that field by the interstage gain.

    Reload the reservoir's frozen start.

    LOOP:

        Remap the scaled field by xor with the pass index.

        Inject that remapping. Step the reservoir.

    GOTO LOOP

    After T passes, the reservoir's live output is the feature field.
    That is what the Readout sees.

The single-stage write-ups live with the siblings. This repository is
the two-stage host.

---

## White noise filter

The Cascade preprocessor behaves as a near unity passthrough at low
to no white noise levels, and offers meaningful filtering effect
at moderate to high noise levels. The write-up is
[`examples/mnist/WhiteNoiseFilter.md`](https://github.com/dliptak001/HypercubeCascade/blob/main/examples/mnist/WhiteNoiseFilter.md).

![MNIST test noise: cascade vs etalon transit vs Bypass](https://raw.githubusercontent.com/dliptak001/HypercubeCascade/main/examples/mnist/cascade_mnist_noise_comp.png)

---

## Raman baseline extraction (a vibrational spectroscopy application)

The first real-world test is Raman spectra: recover the slow
fluorescence background under sharp molecular peaks without
lifting the baseline into the bands or cutting trenches beneath
them. Polynomials, asymmetric least squares, and ordinary
convolutional nets tend to follow the empty stretches well and then
fail where it matters, under peaks and peak clusters. Analysts have
worked around that for decades with spectrum-specific cleanup,
because no method identifies and extracts a true baseline across a
broad range of peak intensities and baseline characteristics
without occasional, and often frequent, human intervention.

The Cascade appears to have solved that problem (albeit on synthetic
data only so far).

Trained for 60 epochs on the LCOHard set — 10,000 synthetic LiCoO₂
(lithium cobalt oxide) spectra — it scores a validation RMSE of
4.82 counts on 2,000 held-out spectra whose baselines span
hundreds of counts.

Below are four held-out validation spectra: grey is the raw
spectrum, red the true baseline, blue the extract. For all four
shown here, and for each of the remaining 1996 validation spectra
not shown, baseline identification is, **WITHOUT EXCEPTION**,
quite remarkable.

And it does this with the thin readout the project aims for: one
HypercubeCNN layer, one convolutional channel, no pooling.

In our judgment this at least matches the best of the established
techniques on spectra like these, and very likely beats them.

![Held-out validation extract, spectra 581 through 584](https://raw.githubusercontent.com/dliptak001/HypercubeCascade/main/examples/RamanBaselineExtraction/extracted_baselines_cascade.png)

### Three hosts, one floor

The etalon-only sibling
([HypercubeEtalon](https://github.com/dliptak001/HypercubeEtalon))
is this Cascade with the reservoir removed. The reservoir-only
sibling ([HypercubeWTF](https://github.com/dliptak001/HypercubeWTF))
is this Cascade with the transit removed: the same frozen reservoir
— same seed, spectral radius, history depth, and pass count — driven
by the normalized spectrum directly. Each of them, on its own,
already does everything described above. With the very same readout
configuration and the same 60-epoch budget, the Etalon scores 4.77
and WTF 4.76 against the Cascade's 4.82, and all three overlays are
indistinguishable from the one shown. Three preprocessors that share
no mechanism — a transit, an orbit, and the two in series — carry
the same one-layer, one-channel readout to the same floor.

Real spectra, however, are not nearly this clean. Low laser power,
short integration times, and weak scatterers all put noise on the
spectrum, and that is where a baseline extractor has to earn its
keep.

That is what the Cascade's second stage, the Reservoir, is for. On
the strength of the MNIST white-noise study
([`examples/mnist/WhiteNoiseFilter.md`](https://github.com/dliptak001/HypercubeCascade/blob/main/examples/mnist/WhiteNoiseFilter.md)),
the Cascade is expected to outperform the Etalon alone in that
noise — and WTF's own study
([`examples/mnist/WhiteNoiseFilter.md`](https://github.com/dliptak001/HypercubeWTF/blob/main/examples/mnist/WhiteNoiseFilter.md))
found that same reservoir a filter that holds accuracy as the noise
rises. Whether the transit in front of it adds anything under noise,
or whether the reservoir is doing all of the filtering, is the open
question.

That is the next experiment.

Side-by-side overlays and all three training profiles are in
[`examples/RamanBaselineExtraction/`](https://github.com/dliptak001/HypercubeCascade/blob/main/examples/RamanBaselineExtraction/README.md).

---

## Installation

**Preferred:** install a pre-built wheel from PyPI (no compiler).

```bash
pip install hypercube-cascade
```

```python
import hypercube_cascade as hc
print(hc.__version__)
```

Package name on PyPI: **`hypercube-cascade`**. Import name:
**`hypercube_cascade`**. Main type: **`hc.Cascade`**.

Wheels target Python 3.10–3.14 on common Windows, Linux, and macOS machines.
Runtime dependency: NumPy only.

### From source (full repository)

To compile the extension yourself, clone this **entire** repository (not a
minimal source-only download of the `python/` folder alone — the C++ core and
vendored HypercubeCNN live next to `python/`). You need Python 3.10+, a C++23
compiler, and CMake ≥ 3.20.

```bash
git clone https://github.com/dliptak001/HypercubeCascade.git
cd HypercubeCascade/python
pip install .
```

On Windows with CLion’s MinGW, put that compiler’s `bin` folder (and Ninja) on
your `PATH`, then:

```bash
pip install . --no-build-isolation --force-reinstall --no-deps
```

(Exact CLion paths change with the version.) Step-by-step toolchain notes:
[docs/Python_SDK.md](https://github.com/dliptak001/HypercubeCascade/blob/main/docs/Python_SDK.md).

---

## Quick start

You bring each sample as a length-**N** float array (N = 2<sup>dim</sup>). How
you get there — pad an image, reshape a spectrum, invent a layout — is up to
you. This package does not pack 784 pixels or 300 bins for you.

Shapes that matter:

| Array | Shape | Notes |
|-------|-------|-------|
| `fields` | `(count, N)` | one length-N field per row |
| `labels` (classification) | `(count,)` | integer class indices |
| `targets` (regression) | `(count, num_outputs)` | float targets |

```python
import numpy as np
import hypercube_cascade as hc

dim = 7
N = 2**dim
rng = np.random.default_rng(0)
fields = rng.standard_normal((200, N), dtype=np.float32)
labels = rng.integers(0, 4, size=200)

cas = hc.Cascade(
    dim=dim,
    exciter_subcube_dim=5,
    history_depth=4,
    T=50,
    ic_seed=2,
    readout_num_outputs=4,
    readout_task="classification",
    readout_epochs=80,
)
cas.fit(fields, labels)  # collect_batch + train

print(cas.N, cas.T, cas.num_collected)
print(f"train sanity check: {cas.accuracy_on_collected():.3f}")
print(cas.predict_class(fields[0]), cas.predict(fields[0]).shape)

cas.save("model.pkl")
loaded = hc.Cascade.load("model.pkl")
```

### Step by step (same loop, more control)

```python
cas = hc.Cascade(
    dim=7,
    exciter_subcube_dim=5,
    readout_num_outputs=4,
    readout_task="classification",
)
cas.collect_batch(fields_train, labels_train)
cas.train()
logits = cas.predict(fields_test[0])       # (num_outputs,) float32
cls = cas.predict_class(fields_test[0])    # int
test_acc = cas.accuracy(fields_test, labels_test)  # held-out, fresh maps
```

For regression, set `readout_task="regression"` and pass float targets instead
of class labels. Then use `r2_on_collected()` / `r2(fields, targets)` the same
way.

`accuracy_on_collected` and `r2_on_collected` only look at the samples you
already trained on — they are a quick sanity check, not a test score. For real
evaluation, hold fields out and call `accuracy` / `r2` (or `predict` /
`predict_class` yourself).

---

## Features

- **One class** — `hypercube_cascade.Cascade` is the whole product surface
- **Map loop** — `collect` / `collect_batch` → `train` →
  `predict` / `predict_class`
- **`fit`** — clear, collect, and train when your arrays are ready
- **dim 5–12** — field length N = 2<sup>dim</sup>; one dim for all three
  stages; orbit length `T`; etalon face `exciter_subcube_dim`
- **Two gains** — `interstage_scale` (transit → orbit) and `readout_scale`
  (orbit → readout)
- **Classification or regression** — `readout_task` fixed at construction
- **Held-out scoring** — `accuracy(fields, labels)` / `r2(fields, targets)`
  map fresh in bulk
- **Bulk calls can parallelize** — `collect_threads` (0 = auto)
- **Inspect a map** — `run(x)` then `last_features()`, plus per-stage probes
  `last_exciter()` / `last_interstage()` / `last_reservoir()` for gain tuning
- **Save / load** — `save` / `load` (pickle: config + readout weights;
  collected samples are not stored). Optional `save_readout_hcnn_model` /
  `load_readout_hcnn_model` for portable HCNW + arch JSON
- **NumPy float32** — arrays converted for you; prefer contiguous float32

---

## Examples

For a first try, paste the [Quick start](#quick-start) after
`pip install hypercube-cascade`. That is self-contained.

If you want a longer walk-through, the demo scripts on GitHub under
[`python/examples/`](https://github.com/dliptak001/HypercubeCascade/tree/main/python/examples)
are there to open or download — they are not added to your machine by pip.

| Script | What it is for |
|--------|----------------|
| [synthetic_classification.py](https://github.com/dliptak001/HypercubeCascade/blob/main/python/examples/synthetic_classification.py) | Multi-class toy fields: `fit`, then train and test accuracy |

```bash
# from a clone of HypercubeCascade, after: pip install hypercube-cascade
python python/examples/synthetic_classification.py
```

These use easy made-up fields so the API is obvious — not scores to publish.
More notes:
[python/examples/README.md](https://github.com/dliptak001/HypercubeCascade/blob/main/python/examples/README.md).

---

## Documentation

| Doc | Role |
|-----|------|
| **[docs/Python_SDK.md](https://github.com/dliptak001/HypercubeCascade/blob/main/docs/Python_SDK.md)** | Canonical Python API — every method, layout, pickle, limits |
| [python/examples/README.md](https://github.com/dliptak001/HypercubeCascade/blob/main/python/examples/README.md) | Demo scripts on GitHub |
| [Project README](https://github.com/dliptak001/HypercubeCascade#readme) | Product story and C++ demos from the repo root |
| [docs/CPP_SDK.md](https://github.com/dliptak001/HypercubeCascade/blob/main/docs/CPP_SDK.md) | Native library guide (same product, C++) |
| [docs/CascadeWhitePaper.md](https://github.com/dliptak001/HypercubeCascade/blob/main/docs/CascadeWhitePaper.md) | The two-stage concept, mechanism by mechanism |
| [WhiteNoiseFilter.md](https://github.com/dliptak001/HypercubeCascade/blob/main/examples/mnist/WhiteNoiseFilter.md) | Early white-noise study (MNIST as a test bed) |

---

## Ecosystem

- **[HypercubeCascade](https://github.com/dliptak001/HypercubeCascade)**: two frozen stages in series plus a thin readout.
- **[HypercubeCNN](https://github.com/dliptak001/HypercubeCNN)**: cube-native conv stack with shared kernels.
- **[HypercubeESN](https://github.com/dliptak001/HypercubeESN)**: echo-state reservoir computing on streams.
- **[HypercubeEtalon](https://github.com/dliptak001/HypercubeEtalon)**: frozen etalon transit plus a thin readout.
- **[HypercubeHopfield](https://github.com/dliptak001/HypercubeHopfield)**: Hopfield-style dynamics on the cube.
- **[HypercubeLCN](https://github.com/dliptak001/HypercubeLCN)**: the locally connected net, every weight trained.
- **[HypercubeWorldModel](https://github.com/dliptak001/HypercubeWorldModel)**: frozen WTF encoder + trained LCN predictor and decoder.
- **[HypercubeWTF](https://github.com/dliptak001/HypercubeWTF)**: frozen reservoir orbit plus a thin readout.

---

## License

Apache 2.0. See [LICENSE](https://github.com/dliptak001/HypercubeCascade/blob/main/LICENSE).
