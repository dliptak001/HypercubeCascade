# HypercubeCascade C++ SDK

You place a fixed pattern on the hypercube — an image pack, a spectrum, or any
field you built yourself. HypercubeCascade **runs one etalon transit over that
field, then drives the result through a frozen reservoir** for a short
synthetic orbit, then **trains a small CNN only on the state at the end**. One
class does the whole loop: collect samples, train the head, predict.

You do not need to learn HypercubeEtalon, HypercubeWTF, or HypercubeCNN first.
Link **`HypercubeCascadeCore`**, include **`Cascade.h`**, and work with
**`Cascade`**. Demos and packing helpers are optional recipes; they are not the
product.

This guide matches the public headers for **1.0.x**.

**Who it is for:** anyone embedding the Cascade in a host (collect → train →
predict), and anyone learning the stack with the same API the demos use.

**What you get:** a C++23 static library. Headers sit at the repo root. A
vendored HypercubeCNN builds the trainable readout; hosts usually never call
HCNN themselves.

| Section | |
|---------|--|
| [1. Why explore HypercubeCascade](#1-why-explore-hypercubecascade) | Family role, what two stages buy |
| [2. The big picture](#2-the-big-picture) | Where Cascade sits among Etalon / WTF / CNN |
| [3. One sample](#3-one-sample-step-by-step) | Transit, orbit, spatial→temporal, mechanics |
| [4. Product surface](#4-what-is-the-product-and-what-is-not) | Headers, rules, the loop |
| [5. Build](#5-build-and-consume) | CMake, binaries |
| [6. First program](#6-first-program) | Minimal collect → train → predict |
| [7. API](#7-the-api-you-actually-use) | Config and methods |
| [8–13](#8-please-do-not) | Boundaries, demos, pitfalls, cheat sheet |

---

## 1. Why explore HypercubeCascade

[HypercubeEtalon](https://github.com/dliptak001/HypercubeEtalon) processes
spatial data through **one frozen preprocessing stage**: an etalon transit —
a deterministic wave swept across every vertex/antipode cavity of the cube.

[HypercubeWTF](https://github.com/dliptak001/HypercubeWTF) processes spatial
data through **a different frozen preprocessing stage**: a short synthetic
orbit on a frozen recurrent reservoir.

HypercubeCascade runs **both, in series, on one cube**: transit first, orbit
second, and the same kind of small HypercubeCNN readout at the end. The
convolution engine never sees the original field; it sees the reservoir's
end state of an orbit driven by the transit of that field.

The point of the experiment is to see whether two preprocessing stages in
front of the CNN outperform the CNN by itself, and outperform either stage
alone. On the MNIST white-noise study the answer so far is yes: near-unity
passthrough on clean fields, and from σ = 0.3 upward the cascade leads the
single-stage transit, with both ahead of the bypass. See
[WhiteNoiseFilter.md](../examples/mnist/WhiteNoiseFilter.md) for the numbers
and [CascadeWhitePaper.md](CascadeWhitePaper.md) for the full mechanism.

The aim is a preprocessor effective enough that the readout can be a single
convolutional layer with a single channel and no pooling — fast to train,
small in memory, and requiring essentially no CNN architecture engineering.
The Raman baseline example already runs at exactly that readout size.

Whether the two-stage pipeline has real product value is still an open
question.

---

## 2. The big picture

Most learning systems either see a **stream** (one small input every step) or
a **static pattern** (classify an image once). The Cascade takes a static
pattern and manufactures both a wave and a clock out of it:

1. You give it one full-length field on the cube (packed however you like).
2. The **Exciter** runs one etalon transit — same field length in, same out.
3. That transit output, times a gain, is **re-addressed** for `T` reservoir
   passes (a synthetic orbit).
4. It takes a **single snapshot at the end** and trains a CNN on those
   features.

Neither preprocessor ever trains. Only the readout does. That is the whole
product idea.

| Library | You typically feed it… | What runs |
|---------|------------------------|-----------|
| HypercubeEtalon | a **static** length-N field | etalon transit → HCNN |
| HypercubeWTF | a **static** length-N field | reservoir orbit → end state → HCNN |
| **HypercubeCascade** | a **static** length-N field | **transit → orbit → end state → HCNN** |

### Your data does not have to be a power of two

**dim** is the size knob for the whole pipeline: set `CascadeConfig::dim`
(valid **5…12**) and you get **N = 2<sup>dim</sup>** vertices. Construction
stamps that one dim onto the Exciter, the Reservoir, and the Readout — the
nested `dim` fields are not independent knobs. The Cascade always expects a
field of length N.

If your raw data is 784 pixels or 300 bins, **you** map it onto N floats first
(pad, resize, spatial embed, custom layout — your choice). The Cascade does
not invent that map. Demo helpers under `examples/common/` are one
MNIST-oriented recipe; skip them when you pack your own way.

### What freezes vs what learns

| Piece | Trains? |
|-------|---------|
| Exciter neighbor weights | No — drawn once at construct |
| Reservoir weights and bias | No — drawn once at construct |
| Starting state **s0** (full delay line) | No — drawn once from `ic_seed`, reloaded every sample |
| The two gains (`interstage_scale`, `readout_scale`) | No — config scalars you set |
| How you pack domain data into the field | Your problem (outside the Cascade) |
| HCNN readout | **Yes** |

### Episode start state (**s0**)

Every sample's orbit begins by reloading the **same** frozen initial condition
into the full delay line, then setting the pass counter `c` to 0. That way two
maps of the same field (same config) are deterministic, and you never inherit
residual state from the previous sample.

| Rule | Behavior |
|------|----------|
| Size | **`N × M`** floats — one full delay-line worth of state |
| When drawn | **Once**, at `Cascade` construction |
| Seed | **`ic_seed`** — separate from `reservoir.seed` (weights) and `exciter.seed` |
| Distribution | i.i.d. uniform on **[-0.5, 0.5]** over the whole buffer |
| After construct | Immutable for the life of that `Cascade` |
| Each sample | Reload into the live delay line (age-correct load, not a blind mid-rotation overwrite) |
| Pass counter | `c = 0` at sample start |

If you change only `ic_seed`, weights stay the same but the end features
change (different orbit start). If you change only `reservoir.seed`, the
frozen recurrent weights change. If you change only `exciter.seed`, the frozen
transit weights change — and on the tasks measured so far, the results barely
move (seed is not a tuning parameter).

---

## 3. One sample, step by step

Think of one map as: transit the field, reset the reservoir to a known start,
drive for a while, read once.

### Stage 1 — the etalon transit

The Exciter is **not** a reservoir: no leak, no delay line, no orbit. It is a
fixed nonlinear map. At construction it draws one weight per (vertex,
neighbor) pair and never updates them again.

An *etalon* here is a start vertex `r` and its face antipode treated as a
pair of reflectors. The face is the subcube spanned by the low `subcube_dim`
bits — `M_walk = 2^subcube_dim` vertices, with the high bits pinned by `r`.
One transit scales the input once (by `exciter.input_scaling`), then for
every start `r` reloads that scaled field, walks the face out to the antipode
and back, and writes one output sample: the value standing at `r` after the
second visit. Each site update is tanh of a full-star neighbor sum, applied
sequentially — the disturbance propagates through the face and reflects back.
Off-face neighbors are never updated during a walk, so every update also
mixes in unmodified input.

`N` starts → `N` output samples → the transit output is a field with the same
length and vertex indexing as the input.

### Stage 2 — making time when you only have a still picture

A classical reservoir expects a **movie**: fresh input every tick. The Cascade
has a **still** — the transit output, times `interstage_scale`. If you only
shoved that field in once and stepped forever, most of the pattern would be a
one-shot kick.

So the Cascade invents a clock from space. Call the pass counter `c` (starts
at 0 each sample). On pass `c`, every vertex `v` is driven by field sample

```text
d[(v XOR c) & (N − 1)]
```

Read that as: **the numbers in the drive field never change**; you only change
**which number sits on which vertex**. XOR with `c` is a fixed, invertible
shuffle of addresses on the cube. Increment `c`, shuffle again. Geometry (who
is neighbor to whom) and all frozen weights stay put — they do not slide with
`c`. What moves is the **registration** of the field onto the graph.

Do that for `T` passes and the reservoir experiences a **synthetic time
series**: the same transit output seen under `T` successive addressings. The
reservoir itself is orthodox echo-state machinery — frozen random weights on
cube edges, spectral radius rescaled to a target, optional leak and bias, a
delay line of depth `M` — and you never backprop through it.

You still follow reservoir-computing discipline at the end: you do **not**
train on intermediate passes. After the last pass you read **once** — the
newest delay-line slice, times `readout_scale`. That length-N field is the
feature row the CNN sees.

### Mechanics in order

1. Copy the caller's field (the Cascade never writes your buffer).
2. One etalon transit over the copy.
3. Multiply the transit output by `interstage_scale`.
4. Reload the frozen initial condition **s0** into the delay line.
5. For pass `c = 0, 1, …, T−1`: place the scaled field on the cube with
   address offset `c`, one reservoir step.
6. Multiply the reservoir's live output by `readout_scale` → features
   (length N).
7. Hand those features to the readout (collect, train, or predict).

```text
x  (length N, fixed for this sample)
    │
    ▼
 Etalon transit → × interstage_scale → load s0 → drive T times
    │
    ▼
 live end state → × readout_scale → features (N)
    │
    ▼
 HCNN readout → class logits or regression values
```

### Words you will see in the API

| Word | Plain meaning |
|------|---------------|
| **dim** | Cube dimension you choose (5…12); one knob for all three stages |
| **N** | Field length = 2<sup>dim</sup> (input, transit output, features — all N) |
| **subcube_dim** | Face size of one etalon cavity; walk covers 2<sup>subcube_dim</sup> vertices |
| **T** | How many drive passes (must be ≥ 1; no auto value) |
| **M** | Delay-line depth (`reservoir.history_depth`) |
| **s0** | Frozen start state, length `N × M`, U[-0.5, 0.5] from `ic_seed`; reloaded every sample |
| **interstage_scale** | Gain between transit output and reservoir drive |
| **readout_scale** | Gain between reservoir end state and readout |

Unlike HypercubeWTF there is no `B` / `readout_slices` knob: the readout
always sees exactly the newest slice, and the feature size is always **N**.

---

## 4. What is the product (and what is not)

| You care about… | Use… |
|-----------------|------|
| Integrating the library | **`Cascade`** + **`CascadeConfig`** |
| Realized spectral radius, saving readout weights | `cas.reservoir()` / `cas.readout()` |
| Per-stage field probes | `cas.LastExciter()` / `LastInterstage()` / `LastReservoir()` / `LastFeatures()` |
| Learning by example | `cascade_synth`, `cascade_mnist`, `cascade_raman` |
| MNIST paths / packing demos | `examples/common/` (optional) |
| Raw HypercubeCNN | Almost never — that lives under the readout |

```text
Cascade.h          front door (Cascade)
CascadeConfig.h    CascadeConfig (nests the three stage configs)
Exciter.h          ExciterConfig (+ Exciter for inspection)
Reservoir.h        ReservoirConfig (+ Reservoir for inspection)
Readout.h          ReadoutConfig, enums, Readout
… .cpp files …
third_party/HypercubeCNN/    vendored; see VENDORED.md
examples/                    demos, not the SDK definition
docs/CPP_SDK.md              this guide
docs/CascadeWhitePaper.md    the concept, in depth
```

Link **`HypercubeCascadeCore`** (it pulls **`HypercubeCNNCore`** for you).

### Rules that matter

These are product contracts, not implementation trivia.

- **Every field is length N.** Wrong size throws.
- **Values are usually kept in [-1, 1].** The library trusts the host; it does not clamp.
- **You pack; the Cascade maps.** No built-in image layout.
- **One dim.** Construction stamps `cfg.dim` onto all three stages; do not
  size them separately.
- **Exciter, Reservoir, and s0 freeze at construct.**
- **`T` must be ≥ 1.** There is no `0 = auto` convention here.
- **Both gains must be finite and > 0.** Default 1 (passthrough).
- **Only the end state** goes to the readout — newest slice, length N.
- **`Predict` returns raw logits** (or regression values) — no softmax.
- **`AccuracyOnCollected` / `R2OnCollected` are training-set scores.** For
  held-out data use `Accuracy(fields, labels)` / `R2(fields, targets)`, which
  map fresh.
- **There is no built-in train-input noise knob.** If a study needs noisy
  collect or noisy eval, the host adds the noise (see `cascade_mnist`).
- **One `Cascade` per thread of control.** Bulk collect parallelizes *inside*
  one call; do not call public methods concurrently on the same object.
- **`Cascade` is not copyable and not movable.** Heap-allocate if it must
  change hands.

### The loop you will write

```text
fill CascadeConfig
construct Cascade once
collect many samples      (Collect / CollectBatch)
TrainOnCollected
Predict / PredictClass    (always a fresh clean map)
```

Optional extras: `Run` + the per-stage `Last*` probes, train-set metrics,
held-out `Accuracy` / `R2`, `ClearCollected`, `collect_threads` for faster
bulk collect.

---

## 5. Build and consume

You need **C++23** and **CMake ≥ 3.21**. Prefer **Release** when you care
about study numbers (Debug and Release float behavior can differ with this
project's fast-math flags).

In CLion: open the project, reload CMake, build. From a shell with the
toolchain available:

```bash
cmake --build cmake-build-release
```

When this repo is the top-level project you also get:

| Binary | Role |
|--------|------|
| `HypercubeCascade` | Link smoke (prints a banner) |
| `cascade_synth` | Multi-class synthetic fields (no data files) |
| `cascade_mnist` | MNIST recipe + test-noise sweep (IDX files under `C:\HypercubeCascade\data`) |
| `cascade_raman` | Raman baseline regression (spectra under `C:\HypercubeCascade\RamanSpectraLCOHard`) |
| `cascade_raman_extract` | Writes selected baseline extracts from a saved readout |

If you pull HypercubeCascade in as a **subdirectory**, demos are skipped; you
still get the library.

```cmake
add_subdirectory(path/to/HypercubeCascade)
add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE HypercubeCascadeCore)
```

```cpp
#include "Cascade.h"
```

(`HypercubeCascadeCore` exports the repo root as a public include directory,
so the one include line is enough.)

---

## 6. First program

A tiny two-class example — collect, train, predict. (Verified against the
library: trains and predicts correctly on this toy task.)

```cpp
#include "Cascade.h"
#include <cstdio>
#include <vector>

int main() {
    CascadeConfig cfg;
    cfg.dim = 5;                        // N = 32 (stamped onto all stages)
    cfg.T = 50;
    cfg.interstage_scale = 1.0f;
    cfg.ic_seed = 2;

    cfg.exciter.subcube_dim = 4;        // default 6 is illegal at dim 5
    cfg.exciter.seed = 1;
    cfg.exciter.input_scaling = 1.0f;
    cfg.exciter.weight_scaling = 0.15f;

    cfg.reservoir.seed = 3;
    cfg.reservoir.spectral_radius = 0.9f;
    cfg.reservoir.history_depth = 4;

    cfg.readout.num_outputs = 2;
    cfg.readout.task = ReadoutTask::Classification;
    cfg.readout.epochs = 80;
    cfg.readout.conv_channels = 4;
    cfg.readout.num_threads = 1;

    Cascade cas(cfg);
    const size_t N = cas.N();

    auto field = [&](int label) {
        std::vector<float> x(N, 0.f);
        const float s = (label == 0) ? 1.f : -1.f;
        for (size_t i = 0; i < N / 2; ++i)
            x[i] = s * (0.2f + 0.8f * float(i) / float(N));
        return x;
    };

    for (int i = 0; i < 24; ++i) {
        cas.Collect(field(0), 0);
        cas.Collect(field(1), 1);
    }
    cas.TrainOnCollected();

    // AccuracyOnCollected is the *training* set, not test data.
    std::printf("train acc=%.3f  pred0=%d pred1=%d\n",
                cas.AccuracyOnCollected(),
                cas.PredictClass(field(0)),
                cas.PredictClass(field(1)));
    return 0;
}
```

**Habits that save pain later**

1. Build the config, construct **one** `Cascade` (weights and s0 freeze here).
2. Collect a dataset, then train. You can call `TrainOnCollected` again
   without clearing — it continues from the current readout weights.
3. Treat `Predict` / `PredictClass` as clean inference — each is a fresh map.
4. Keep packing in your code (or a demo helper). The core only accepts
   length-N fields.

---

## 7. The API you actually use

Authoritative signatures and contracts live in **`Cascade.h`** and
**`CascadeConfig.h`** (and the headers they pull). This section is the
host-oriented map.

### Config at a glance

Everything interesting is set **before** `Cascade` is constructed.

```cpp
struct CascadeConfig {
    size_t dim = 8;                  // one cube dim for all three stages [5, 12]

    ExciterConfig   exciter{};
    ReservoirConfig reservoir{};
    ReadoutConfig   readout{};

    size_t collect_threads = 0;      // bulk maps: 0 = auto, 1 = serial, K = K
    size_t T = 100;                  // drive passes; must be >= 1
    float  interstage_scale = 1.0f;  // transit → reservoir gain (finite, > 0)
    float  readout_scale = 1.0f;     // reservoir → readout gain (finite, > 0)
    uint64_t ic_seed = 1;            // s0 only (not a weight seed)
};
```

**Exciter** (frozen transit) — the stage-1 knobs:

| Field | Meaning | Valid / notes |
|-------|---------|---------------|
| `seed` | Neighbor-weight draws | any `uint64_t`; results are seed-insensitive |
| `input_scaling` | Scales the field once before the transit | demos use 0.1…2.0 |
| `weight_scaling` | Neighbor weights are U(-1, 1) × this | demos use 0.1…0.15 |
| `subcube_dim` | Etalon face size; walk covers 2<sup>subcube_dim</sup> vertices | **[1, dim]** — the default 6 throws below dim 6 |

**Reservoir** (frozen dynamics) — common knobs. Header defaults exist; in-tree
demos tune these widely, so treat "typical" as a starting band, not a recipe.

| Field | Meaning | Valid / notes | Often in demos |
|-------|---------|---------------|----------------|
| `seed` | Weight draws | any `uint64_t` | fixed per experiment |
| `spectral_radius` | Target for recurrent rescale | > 0 | 0.9…0.98 |
| `leak_rate` | Mix each step | (0, 1] | 0.9…1 |
| `input_scaling` | How hard the drive field injects | ≥ 0 | 0.02…1.0 |
| `history_depth` | M (delay-line depth) | 1…64 | 2…4 |
| `bias_scaling` | Bias strength; 0 = off | ≥ 0 | 0 |
| `verbose` | Construction printout | bool | false |

(`exciter.dim` and `reservoir.dim` exist on the nested structs but the host
overwrites them with `cfg.dim` — leave them alone.)

**Readout** (trainable head) — knobs most hosts touch:

| Field | Meaning |
|-------|---------|
| `num_outputs` | Classes, or regression width |
| `task` | `ReadoutTask::Classification` or `Regression` |
| `epochs`, `batch_size` | Batch training |
| `lr_max`, `lr_min_frac`, `lr_decay_epochs` | Cosine learning-rate schedule |
| `num_layers`, `conv_channels`, `channel_growth` | Conv stack size (the goal is 1 / 1 / 1) |
| `use_pooling`, `pool_type` | Antipodal pool after each conv |
| `activation` | Per-conv activation (demos often `NONE`) |
| `num_threads` | HCNN workers — use **1** for simple determinism |
| `restore_best_epoch` | Keep best-epoch weights (default true) |
| `seed` | Readout weight init |

Deeper fields (batch-norm, optimizer, holdout fraction, `epoch_tick`) live on
`ReadoutConfig` in `Readout.h`.

### After construct — sizes and inspection

```cpp
explicit Cascade(const CascadeConfig& cfg);

cas.Dim();  cas.N();
cas.NumCollected();
cas.CollectedFeatures();   // span, sample-major, NumCollected() * N
cas.NumOutputs();
cas.CollectThreads();      // configured preference (0 = auto)

cas.config();              // resolved knobs (dim stamped onto stages)
cas.exciter();             // const — transit config, walk size
cas.reservoir();           // const — e.g. realized spectral radius
cas.readout();             // mutable — weights, HCNW save/load, IsTrained
```

Construction checks the usual mistakes: dim out of [5, 12], `subcube_dim` out
of [1, dim], `T < 1`, non-finite or non-positive gains, `num_outputs < 1`,
plus every stage's own checks.

### Run a map (no training)

```cpp
cas.Run(x);                        // x.size() == N; x is not modified
auto f  = cas.LastFeatures();      // length N — what the readout would see
auto y  = cas.LastExciter();       // transit output
auto d  = cas.LastInterstage();    // transit × interstage_scale (the drive)
auto z  = cas.LastReservoir();     // end state before readout_scale
```

The three stage probes (`LastExciter` / `LastInterstage` / `LastReservoir`)
update on **serial** maps only (`Run`, `Collect`, `Predict`, `PredictClass`).
`LastFeatures` updates on every completed map, including bulk calls (last row).
All spans are valid until the next map on the instance — copy what you keep.

The demos use these probes to print per-stage mean-|value| lines ("~1 is a
live field, ~0 is crushed") — the fastest way to tune the two gains.

### Collect, train, predict

**Classification**

```cpp
cas.Collect(x, class_label);              // one sample
cas.CollectBatch(fields_flat, labels);    // bulk, sample-major
```

**Regression** — same idea with target vectors (`num_outputs` floats per
sample):

```cpp
cas.Collect(x, targets);
cas.CollectBatch(fields_flat, targets_flat);
```

```cpp
cas.ClearCollected();            // drop training rows (keeps worker pool)
cas.TrainOnCollected();          // needs at least one sample; does not clear

auto out = cas.Predict(x);       // num_outputs floats; no softmax
int y = cas.PredictClass(x);     // classification only

double acc = cas.AccuracyOnCollected();   // train set
double r2  = cas.R2OnCollected();         // train set, regression
```

**Held-out scoring** — these map every field fresh (bulk, parallel), then
score:

```cpp
double test_acc = cas.Accuracy(test_fields_flat, test_labels);
double test_r2  = cas.R2(test_fields_flat, test_targets_flat);
```

Bulk layout notes:

- `fields_flat` is sample-major: sample `i` starts at `i * N`.
- Labels are validated **before** any mapping starts; a throw leaves the
  collected set unchanged.
- Wrong task (class API on a regression net, etc.) throws.

### Faster bulk collect

`CascadeConfig::collect_threads`:

- `0` — auto (leaves one or two cores free so the machine stays responsive)
- `1` — serial
- `K` — up to K workers

Worker 0 reuses the primary Exciter and Reservoir. Extra workers clone the
frozen stages once (same seeds → identical weights). The internal thread pool
**grows** for the life of the `Cascade` and does not shrink. Single-sample
calls are always serial.

---

## 8. Please do not

| Temptation | Better path |
|------------|-------------|
| Drive `Exciter` / `Reservoir` yourself for product training | Use `Cascade` maps |
| Call vendored `hcnn::HCNN` from the app | Let `Readout` own it; export via `cas.readout()` if needed |
| Depend on `examples/common` in production | Copy the idea; own your packing |
| Stream intermediate passes into the readout | Product samples **end of orbit only** |
| Reorder the stages (orbit → transit) | Not a knob; it is an open experiment, not a config |
| Size the three stages separately | One `dim`; construction stamps it |

`Exciter`, `Reservoir`, and `Readout` headers are public so config and
inspection work. The happy path is still collect → train → predict on
**`Cascade`**.

---

## 9. Demos as recipes

Demos keep product knobs in `MakeBaseConfig()` and demo-only constants (`k*`)
beside them:

```text
config → Cascade → collect → train → score → predict
```

| Demo | When to open it |
|------|-----------------|
| `examples/synth/cascade_synth.cpp` | Fast multi-class gate without data files |
| `examples/mnist/cascade_mnist.cpp` | Real packing, stage-scale probes, test-noise sweep |
| `examples/RamanBaselineExtraction/cascade_raman.cpp` | Regression with the minimal readout, save + reload check |
| `examples/RamanBaselineExtraction/cascade_raman_extract.cpp` | Inference-only from a saved readout |

More context: [`examples/README.md`](../examples/README.md).

---

## 10. Threads, memory, and cost

- Treat one `Cascade` as **exclusive** for public calls.
- Bulk collect is where parallelism belongs; it clones the frozen stages as
  needed.
- Per-sample cost has two parts: the transit
  (`N × (2 × 2^subcube_dim − 1) × dim` multiply-adds, plus an N-float reload
  per start) and the orbit (`T × N × dim × (history_depth + 1)`). In the
  shipped configs the orbit is 2–4× the transit.
- Features are always `N` floats; the CNN scales with its own layers and
  channels.
- If you already run many `Cascade` instances in parallel, set
  `readout.num_threads = 1` so HCNN does not oversubscribe the machine.
- Prefer Release when comparing accuracies across runs.

---

## 11. Common mistakes

| Symptom / assumption | Fix |
|----------------------|-----|
| Throw on collect / run | Field length must equal `cas.N()` |
| "Why can't I pass 784 floats?" | Pack to N first |
| Construct throws at small dim | Default `exciter.subcube_dim = 6` needs dim ≥ 6 — set it ≤ dim |
| Construct throws on T | `T` must be ≥ 1; there is no `0 = auto` here |
| Construct throws on a gain | `interstage_scale` / `readout_scale` must be finite and > 0 |
| Softmax inside `Predict` | You get logits; use `PredictClass` or argmax |
| Expected a train-noise σ knob | Not in this host — add noise in your collect loop |
| Stage probes empty after bulk collect | `LastExciter` / `LastInterstage` / `LastReservoir` update on serial maps only |
| Great train accuracy, bad real test | `AccuracyOnCollected` is the training set; use `Accuracy(...)` on held-out fields |
| Features look crushed (~0) | Probe the stages with `Run` + `Last*`; tune `interstage_scale` / `readout_scale` |
| Racey results with shared `Cascade` | One instance, one host thread of control |
| Linked HypercubeCNN only | Link `HypercubeCascadeCore`, include `Cascade.h` |

---

## 12. Further reading

| Doc | What it is |
|-----|------------|
| [CascadeWhitePaper.md](CascadeWhitePaper.md) | The two-stage concept, mechanism by mechanism |
| [WhiteNoiseFilter.md](../examples/mnist/WhiteNoiseFilter.md) | White-noise study: cascade vs transit vs bypass |
| [RamanBaselineExtraction/README.md](../examples/RamanBaselineExtraction/README.md) | Regression task, cascade vs etalon-only comparison |
| [examples/README.md](../examples/README.md) | Demo map and data-file notes |
| [VENDORED.md](../third_party/HypercubeCNN/VENDORED.md) | Which HypercubeCNN pin is in tree |
| [HypercubeEtalon](https://github.com/dliptak001/HypercubeEtalon) / [HypercubeWTF](https://github.com/dliptak001/HypercubeWTF) | The single-stage siblings and their write-ups |

---

## 13. Cheat sheet

```text
#include "Cascade.h"

CascadeConfig cfg;
cfg.dim = 7;                           // N = 128, one knob for all stages
cfg.T = 50;
cfg.interstage_scale = 1.0f;
cfg.readout_scale = 1.0f;
cfg.ic_seed = 1;
cfg.collect_threads = 0;               // auto

cfg.exciter.subcube_dim = 5;           // <= dim
cfg.exciter.input_scaling = 1.0f;
cfg.exciter.weight_scaling = 0.15f;

cfg.reservoir.spectral_radius = 0.9f;
cfg.reservoir.history_depth = 4;

cfg.readout.num_outputs = K;
cfg.readout.task = ReadoutTask::Classification;
cfg.readout.epochs = 100;
cfg.readout.num_threads = 1;

Cascade cas(cfg);

cas.CollectBatch(fields_flat, labels);      // count * N floats, sample-major
cas.TrainOnCollected();

int y = cas.PredictClass(x);
auto logits = cas.Predict(x);
double test_acc = cas.Accuracy(test_flat, test_labels);

cas.Run(x);                                 // probe one map
auto feats = cas.LastFeatures();            // N
float sr = cas.reservoir().GetRealizedSpectralRadius();
```

**In one line:** pack a field → frozen transit → frozen orbit → end features →
train the CNN head.
