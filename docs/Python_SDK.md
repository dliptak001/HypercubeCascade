# HypercubeCascade Python SDK

Static fields have no natural clock. HypercubeCascade preprocesses each
length-N field with **two frozen hypercube stages** — one etalon transit, then
a short reservoir orbit — and trains a small
[HypercubeCNN](https://github.com/dliptak001/HypercubeCNN) readout only on the
**end state**. One class — `Cascade` — owns collect → train → predict.

This is a **map API**, not a stream API. There is no per-tick input sequence
and no next-step `fit` on a 1D signal (that is
[HypercubeESN](https://github.com/dliptak001/HypercubeESN)). Time here is
synthetic and **per sample**.

C++ core and contracts: **[CPP_SDK.md](CPP_SDK.md)**.  
The concept in depth: **[CascadeWhitePaper.md](CascadeWhitePaper.md)**.  
PyPI-facing package story: **[python/README.md](../python/README.md)**.  
Package version: single source `python/hypercube_cascade/_version.py`
(`hypercube_cascade.__version__` and wheel metadata both read it).

## Contents

- [Installation](#installation)
- [Quick start](#quick-start)
- [What a map is](#what-a-map-is)
- [Pipeline vocabulary](#pipeline-vocabulary)
- [API reference](#api-reference)
- [Input data layout](#input-data-layout)
- [Data types](#data-types)
- [Error handling](#error-handling)
- [Model persistence](#model-persistence)
- [Limitations](#limitations)
- [Dependencies](#dependencies)

## Installation

### From PyPI (preferred)

Pre-built **wheels** — no compiler required:

```bash
pip install hypercube-cascade
```

Import as `import hypercube_cascade as hc` (PyPI name `hypercube-cascade`).
Wheels cover Python 3.10–3.14 on common Windows (x64), Linux (x86_64,
aarch64), and macOS (x86_64, arm64) builds. NumPy is the only runtime
dependency.

### From source (full repository)

Compile only from a **full clone** of HypercubeCascade. The extension links
the C++ core and vendored HypercubeCNN that sit **outside** the `python/`
package directory; a `python/`-only tree is not enough.

Requirements: Python 3.10+, C++23 compiler (GCC 13+, Clang 17+, MSVC 2022+),
CMake 3.20+, scikit-build-core, pybind11, NumPy.

```bash
git clone https://github.com/dliptak001/HypercubeCascade.git
cd HypercubeCascade/python
pip install .
```

On Windows with MinGW (e.g. CLion toolchain):

```powershell
pip install scikit-build-core pybind11 numpy
$env:PATH = "C:\path\to\mingw\bin;" + $env:PATH
$env:CMAKE_GENERATOR = "Ninja"
$env:CMAKE_MAKE_PROGRAM = "C:\path\to\ninja.exe"
$env:CC = "C:\path\to\mingw\bin\gcc.exe"
$env:CXX = "C:\path\to\mingw\bin\g++.exe"
pip install . --no-build-isolation
```

### Running tests

From the `python/` directory after install:

```bash
pip install ".[test]"
pytest tests/ -v --import-mode=importlib
```

Or from the repository root: `pytest python/tests/ -v --import-mode=importlib`.
Importlib mode avoids the source tree shadowing the installed `_core`
extension. Use the `pytest` entry point, not `python -m pytest` — the latter
puts the current directory on `sys.path`, and from `python/` the source
package (which has no compiled `_core`) then shadows the installed one.

### Examples

The [Quick start](#quick-start) below is enough after `pip install`. Longer
demos live in the **git tree** under
[`python/examples/`](../python/examples/README.md) — they are **not** part of
the wheel. From a clone, repository root:

```bash
pip install hypercube-cascade   # or: pip install ./python
python python/examples/synthetic_classification.py
```

## Quick start

```python
import numpy as np
import hypercube_cascade as hc

dim = 7
N = 1 << dim
rng = np.random.default_rng(0)
fields = rng.standard_normal((128, N), dtype=np.float32)
labels = rng.integers(0, 4, size=128, dtype=np.int32)

cas = hc.Cascade(
    dim=dim,
    exciter_subcube_dim=5,
    ic_seed=2,
    readout_num_outputs=4,
    readout_task="classification",
    readout_epochs=80,
)
cas.fit(fields, labels)

print(cas.accuracy_on_collected())  # train-set only — not a test score
print(cas.predict_class(fields[0]))
```

### Explicit (full control)

```python
cas = hc.Cascade(
    dim=6,
    exciter_subcube_dim=5,
    history_depth=4,
    T=50,
    interstage_scale=5.5,
    readout_num_outputs=3,
    readout_task="classification",
)
cas.collect_batch(fields_train, labels_train)
cas.train()
logits = cas.predict(fields_test[0])   # shape (num_outputs,)
cls = cas.predict_class(fields_test[0])
test_acc = cas.accuracy(fields_test, labels_test)  # held-out, fresh maps
```

`fit` is `clear_collected` (optional) → `collect_batch` → `train`. Prefer
`fit` for a first pass; use collect/train when you append batches or retrain
without re-mapping every field.

## What a map is

```text
x  (length-N field, host-packed)
    │
    ▼
 etalon transit  →  × interstage_scale  →  reload frozen IC
    │
    ▼
 drive T re-addressed passes  →  end state × readout_scale
    │
    ▼
 features (N)  →  HypercubeCNN  →  logits / values
```

- **N = 2^dim** vertices / field length (dim 5…12; one dim for all stages).
- Exciter and reservoir weights are **frozen** after construction; only the
  readout trains.
- **Predict** always runs a **fresh** map.
- Host packing (MNIST → N, spectra → N, …) is **your** problem — this package
  does not reshape domain data onto the cube.

The CNN head never sees the original field; it sees what the transit and the
orbit leave behind.

## Pipeline vocabulary

| Term | Meaning |
|------|---------|
| **Field** | Length-N float32 vector on the cube (you pack domain data) |
| **Transit** | One frozen etalon sweep: field in, same-length field out |
| **Map** | Transit → gain → reload frozen IC → T orbit passes → end features |
| **Collect** | Run a map → append features + label/target |
| **Train** | Batch-train HCNN on all collected samples |
| **Predict** | Fresh map + readout forward |
| **N** | Vertices / field length = 2^dim |
| **M** | `history_depth` — delay-line depth |
| **subcube_dim** | Etalon face size; one walk covers 2^subcube_dim vertices |
| **T** | Drive-pass count per map (**must be ≥ 1** — no auto value) |
| **interstage_scale / readout_scale** | The two gains between the stages |

Unlike HypercubeWTF there is no `readout_slices` (B) knob and no train-noise
or bypass knob: features are always the newest slice (length N), and noise or
bypass comparisons are the host's job.

## API reference

### Constructor `Cascade(dim, **kwargs)`

All knobs are fixed at construction (same contract as C++ `CascadeConfig`).
One `dim` is stamped onto the Exciter, the Reservoir, and the Readout.

```python
import hypercube_cascade as hc

cas = hc.Cascade(
    dim=7,                          # required; 5–12; N = 2^dim
    T=100,                          # orbit passes; must be >= 1
    interstage_scale=1.0,           # transit → orbit gain (finite, > 0)
    readout_scale=1.0,              # orbit → readout gain (finite, > 0)
    ic_seed=1,                      # frozen episode IC (not a weight seed)
    collect_threads=0,              # 0 = auto
    exciter_seed=7934791766227647176,
    exciter_input_scaling=0.02,
    exciter_weight_scaling=0.02,
    exciter_subcube_dim=6,          # [1, dim] — set <= dim when dim < 6!
    reservoir_seed=7934791766227647176,
    spectral_radius=0.999,
    reservoir_input_scaling=0.02,
    leak_rate=1.0,
    history_depth=16,               # M
    bias_scaling=0.003,
    verbose=False,
    readout_num_outputs=1,
    readout_task="regression",      # or "classification"
    # … readout_* kwargs below
)
```

#### Cascade, exciter, and reservoir

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dim` | `int` | required | Hypercube dimension **[5, 12]**. N = 2^dim, all stages. |
| `T` | `int` | `100` | Orbit passes per map. **Must be ≥ 1** (no `0 = auto`). |
| `interstage_scale` | `float` | `1.0` | Gain on the transit output before the orbit. Finite, > 0. |
| `readout_scale` | `float` | `1.0` | Gain on the orbit end state before the readout. Finite, > 0. |
| `ic_seed` | `int` | `1` | Frozen episode IC seed (separate from both weight seeds). |
| `collect_threads` | `int` | `0` | Bulk workers: 0 = auto, 1 = serial, K = K workers. |
| `exciter_seed` | `int` | `7934791766227647176` | Exciter weight-init seed (matches C++). |
| `exciter_input_scaling` | `float` | `0.02` | Scalar applied once to the field before the transit. |
| `exciter_weight_scaling` | `float` | `0.02` | Exciter neighbor weights are U(-1, 1) × this. |
| `exciter_subcube_dim` | `int` | `6` | Etalon face size; walk covers 2^subcube_dim vertices. Valid **[1, dim]** — the default 6 is rejected below dim 6. |
| `reservoir_seed` | `int` | `7934791766227647176` | Reservoir weight-init seed (matches C++). |
| `spectral_radius` | `float` | `0.999` | Target spectral radius for recurrent weights. |
| `reservoir_input_scaling` | `float` | `0.02` | Reservoir input drive coefficient. |
| `leak_rate` | `float` | `1.0` | Leaky integrator; 1.0 = full replacement. |
| `history_depth` | `int` | `16` | Delay-line depth **M ∈ [1, 64]**. |
| `bias_scaling` | `float` | `0.003` | Per-neuron bias after tanh; 0 disables. |
| `verbose` | `bool` | `False` | Reservoir construction banner. |

**Gain tuning:** the default injection (`interstage_scale=1.0` with
`reservoir_input_scaling=0.02`) is deliberately weak — on many tasks the
features come out crushed toward zero and the readout cannot learn. Probe with
`run(x)` + `last_reservoir()` and raise `interstage_scale` (the in-tree demos
use 1.0–5.5) until the end state is alive.

#### Readout (HCNN)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `readout_num_outputs` | `int` | `1` | Classes (classification) or regression width. |
| `readout_task` | `str` | `"regression"` | `"regression"` or `"classification"`. |
| `readout_num_layers` | `int` | `1` | Conv(+Pool) stages. **`0` = auto** `min(dim−2, 2)`. |
| `readout_conv_channels` | `int` | `16` | Base channel count for the first conv. |
| `readout_epochs` | `int` | `200` | Batch-train epochs. |
| `readout_batch_size` | `int` | `32` | Mini-batch size. |
| `readout_lr_max` | `float` | `0.0015` | Cosine peak LR. Keep ≤ ~0.005 to avoid NaN. |
| `readout_lr_min_frac` | `float` | `0.01` | Floor = `lr_max * lr_min_frac`. |
| `readout_lr_decay_epochs` | `int` | `0` | Cosine horizon; `0` = use `readout_epochs`. |
| `readout_weight_decay` | `float` | `0.0` | L2 on CNN weights. |
| `readout_momentum` | `float` | `0.9` | SGD momentum; ignored under the default Adam optimizer. |
| `readout_activation` | `str` | `"tanh"` | `"tanh"`, `"relu"`, `"leaky_relu"`, or `"none"`. |
| `readout_seed` | `int` | `42` | CNN weight-init seed. |
| `readout_num_threads` | `int` | `0` | HCNN workers: 0 = auto, 1 = single-threaded. |
| `readout_restore_best_epoch` | `bool` | `True` | Restore best-epoch weights after batch train. |
| `readout_best_epoch_holdout_frac` | `float` | `0.0` | Tail hold-out for best-epoch scoring; 0 = full train set. |
| `readout_use_pooling` | `bool` | `True` | Antipodal pool after each conv. |

**Not bound in Python yet** (C++ `ReadoutConfig` only): optimizer choice (C++
default Adam), pool type, channel growth, batch-norm. C++ defaults apply.

### Methods

| Method | Role |
|--------|------|
| `run(x)` | Map one field (no training-set append). Updates `last_features()` and the stage probes. |
| `last_features()` | Length-N float32 from the most recent completed map — updated by **every** map, including bulk calls (last row). |
| `last_exciter()` | Transit output from the last **serial** map (`run`, `collect`, `predict`, `predict_class`). Not updated by bulk calls. |
| `last_interstage()` | Transit × interstage_scale (the orbit drive) — same lifetime as `last_exciter`. |
| `last_reservoir()` | Orbit end state before readout_scale — same lifetime as `last_exciter`. |
| `clear_collected()` | Drop the batch training buffer. |
| `collect(x, target)` | Serial append one sample (label or regression vector). |
| `collect_batch(fields, targets)` | Bulk parallel append. |
| `fit(fields, targets, *, clear=True)` | Optional clear → collect → train. Returns `self`. |
| `train()` | Batch-train HCNN on all collected samples. Does not clear the set. |
| `predict(x)` | Fresh map + forward → shape `(num_outputs,)` float32. |
| `predict_class(x)` | Fresh map + argmax class (classification task only). |
| `accuracy_on_collected()` | Accuracy on the **collected training set** only. |
| `r2_on_collected()` | R² on the **collected training set** only. |
| `accuracy(fields, labels)` | Fresh bulk maps + accuracy on a **held-out** set (classification). |
| `r2(fields, targets)` | Fresh bulk maps + R² on a **held-out** set (regression). |
| `save(path)` / `load(path)` | Pickle constructor config + readout weights. |
| `save_readout_hcnn_model(path_stem)` | Portable `stem.hcnw` + `stem.arch.json`. |
| `load_readout_hcnn_model(path_stem, *, mode="eval")` | Load HCNW into this instance (`"eval"` or `"resume_train"`). |
| `readout_arch_summary()` | Human-readable HCNN architecture and parameter counts. |

### Properties

| Property | Meaning |
|----------|---------|
| `dim`, `N`, `T`, `M` | Geometry and orbit knobs (N = 2^dim, M = history depth) |
| `subcube_dim`, `walk_size` | Etalon face dim and 2^subcube_dim |
| `interstage_scale`, `readout_scale` | The two gains |
| `feature_size` | Floats per sample / `last_features` — always N |
| `num_collected` | Samples in the batch training buffer |
| `num_outputs` | Readout width |
| `exciter_seed`, `reservoir_seed`, `ic_seed` | The three seeds |
| `exciter_input_scaling`, `exciter_weight_scaling` | Exciter config mirrors |
| `spectral_radius`, `realized_spectral_radius` | Target vs post-rescale estimate |
| `reservoir_input_scaling`, `leak_rate`, `history_depth`, `bias_scaling` | Reservoir config mirrors |
| `collect_threads` | Bulk-worker preference (0 = auto) |
| `readout_task` | `"regression"` or `"classification"` |
| `readout_best_epoch` | 1-based best epoch after restore; else 0 |
| `verbose` | Construction banner flag |

## Input data layout

- **Fields** must be length **N** per sample. Prefer shape `(count, N)` for
  bulk APIs; a flat length `count * N` vector is also accepted.
- **Host packing** (images, spectra, sensors → N) is outside this package.
- **Classification labels**: integer class indices in **`[0, num_outputs)`**
  (enforced at collect / scoring). Shape `(count,)` for bulk calls.
- **Regression targets**: shape `(count, num_outputs)` float32 (or flat
  `count * num_outputs`).
- Single-sample methods accept any array that ravel-flattens to the right
  length.

## Data types

| Role | Preferred type | Notes |
|------|----------------|-------|
| Fields / features / predictions | `float32` | Other dtypes converted via NumPy to contiguous float32 |
| Class labels | `int32` (or Python `int`) | Must be in `[0, num_outputs)` (C++ enforces) |
| Bool as a class label | rejected on serial collect | `collect` raises `TypeError`; use an integer index. Bulk `collect_batch` coerces via int32 (do not rely on bool labels). |

## Error handling

Python-side checks raise `ValueError` or `TypeError` with a short message (bad
`dim`, `exciter_subcube_dim`, task string, activation, field shape, label
count, …). Native `std::invalid_argument` maps to `ValueError`; other C++
failures typically surface as `RuntimeError` via pybind11.

Typical mistakes:

- Field length ≠ N
- `exciter_subcube_dim` left at its default 6 with dim 5
- `T = 0` (there is no auto value — C++ requires T ≥ 1)
- `interstage_scale` / `readout_scale` zero, negative, or non-finite
- Bulk `fields` / `targets` row counts disagree
- Class label outside `[0, num_outputs)`
- `predict_class` / `accuracy` on a regression model
- Calling `train` or `accuracy_on_collected` with an empty collected set

## Model persistence

| Mechanism | What is stored | Collected samples? |
|-----------|----------------|---------------------|
| `save` / `pickle` | Constructor config + readout weight blob | **No** (`num_collected` is 0 after load) |
| `save_readout_hcnn_model` | Portable HCNW + arch sidecar | **No** |

Pickle version is bumped when the serialized layout changes; newer libraries
reject unknown future versions with an upgrade message.

```python
cas.save("model.pkl")
cas2 = hc.Cascade.load("model.pkl")  # same ctor knobs + weights; empty collect buffer

cas.save_readout_hcnn_model("export/stem")   # stem.hcnw + stem.arch.json
# Target instance must build a matching HCNN input shape / task (same dim and
# readout_* architecture knobs as the exporter — not only dim/outputs).
cas3 = hc.Cascade(
    dim=cas.dim,
    exciter_subcube_dim=cas.subcube_dim,
    history_depth=cas.history_depth,
    readout_num_outputs=cas.num_outputs,
    readout_task=cas.readout_task,
    # plus any non-default readout_num_layers / channels / pooling / …
)
cas3.load_readout_hcnn_model("export/stem", mode="eval")
```

The preprocessors themselves are never serialized — they reconstruct exactly
from the constructor seeds and scalars. A pickle therefore captures the whole
product: config in, identical frozen stages out, plus the trained readout.

Prefer `save` / `load` when you want a full Python round-trip of the product
config. Prefer HCNW when you need a portable HypercubeCNN weight export.

**Security:** `load` uses `pickle.load`. Never load untrusted files.

## Limitations

- One `Cascade` instance is **not thread-safe** for concurrent public calls
  from multiple host threads. Bulk parallelism is internal only.
- `accuracy_on_collected` / `r2_on_collected` only score samples you already
  collected (and typically trained on). Use `accuracy` / `r2` on held-out
  fields for real evaluation.
- `last_exciter` / `last_interstage` / `last_reservoir` are serial-map probes;
  bulk calls do not update them (`last_features` is updated by every map).
- No train-noise or bypass knob (HypercubeWTF has both); add noise or run
  comparisons in host code.
- A few readout knobs remain C++-only (optimizer, pool type, channel growth,
  batch-norm); see constructor tables above.
- Native contracts, map mechanics, and host integration detail:
  **[CPP_SDK.md](CPP_SDK.md)**.

## Dependencies

| Layer | What |
|-------|------|
| Runtime | NumPy |
| Wheel install | No compiler |
| From-source build | Full repo clone, C++23, CMake ≥ 3.20, scikit-build-core, pybind11 |

The HypercubeCNN readout is built into the extension — no separate HCNN
package.
