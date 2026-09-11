# HypercubeCascade: Two Frozen Hypercube Preprocessors in Series

**A technical white paper on the core concept.**

*Describes the HypercubeCascade codebase as of commit `decea06`. Every
mechanism in this paper is stated from the source in this repository;
every empirical number is quoted from a repository write-up and marked
with its provenance. Nothing here is projected or extrapolated.*

---

## 1. Thesis

HypercubeCascade is an experiment with a single question behind it:

> Can two frozen, untrained preprocessing stages — both native to a
> Boolean hypercube — transform a spatial input field so effectively
> that the trained network behind them can be reduced to almost
> nothing?

The pipeline is:

```
x[N] ──▶ Exciter ──▶ y[N] ──▶ × interstage_scale ──▶ Reservoir (T passes)
     ──▶ z[N] ──▶ × readout_scale ──▶ Readout (HypercubeCNN)
```

The **Exciter** runs one *etalon transit* — a deterministic wave-like
sweep over every reflective vertex/antipode cavity of the cube
(the mechanism of the sibling project HypercubeEtalon). The
**Reservoir** drives a short *synthetic orbit* on a frozen recurrent
hypercube network (the mechanism of the sibling project HypercubeWTF).
The **Readout** is a small HypercubeCNN that is the only trained
component in the system.

Both preprocessors are frozen at construction: their weights are drawn
once from seeded RNG streams and never updated. Only the Readout
learns. The stated aim (root `README.md`) is a preprocessor effective
enough that the readout can be **a single convolutional layer with a
single channel and no pooling** — fast to train, small in memory, and
requiring essentially no CNN architecture engineering.

The experimental comparison the project exists to make: does
Exciter → Reservoir → Readout outperform (a) the Readout alone, and
(b) either preprocessor alone in front of the same Readout?

---

## 2. The substrate: one Boolean hypercube

All three stages live on the same cube. A single `dim` in
`CascadeConfig` (valid range 5–12) is stamped onto the Exciter, the
Reservoir, and the Readout at construction. The field length is
`N = 2^dim` floats; a vertex is an integer index `v` in `[0, N)`; the
`dim` nearest neighbors of `v` are `v ^ (1 << j)` for
`j = 0 … dim-1` — one bit flip each.

Properties the project builds on (root `README.md`):

- **The graph is never stored.** Connectivity is implicit in the
  vertex indices. A stage reconstructs exactly from a seed and a few
  config scalars.
- **Perfect homogeneity.** Every vertex has the same degree and the
  same local neighborhood structure — no structural favorites baked in
  by a random graph.
- **Cheap navigation.** A neighbor is a bit operation on the index,
  not a pointer chase.
- **Topology-native pairing.** The Readout (a hypercube CNN) consumes
  the Reservoir's output field with zero geometric distortion; the
  data never leaves the cube it was generated on.

Getting external data onto the cube is the caller's job. The examples
show two patterns: 1-D signals whose length is already `N` map
bin-per-vertex with no packing (Raman, `N = 2048`, dim 11), and 2-D
images pack through HypercubeCNN's spatial embedder (MNIST 28×28 into
`N = 1024`, dim 10: the full digit in low addresses plus a centered
crop filling the remaining budget).

---

## 3. Stage 1: the Exciter (one etalon transit)

Source: `Exciter.h` / `Exciter.cpp`. Byte-identical to the Exciter in
HypercubeEtalon.

### 3.1 Frozen state

At construction the Exciter draws one weight per (vertex, neighbor)
pair — `N · dim` floats — from `U(-1, 1) × weight_scaling`, using a
single `mt19937_64` stream seeded with `cfg.seed`. These weights are
never trained. There is no leak, no delay line, and no orbit: the
Exciter is a fixed nonlinear map, not a reservoir.

Its own `dim` range is 4–12 (the Cascade host narrows the shared dim
to 5–12); `subcube_dim` must lie in `[1, dim]`.

### 3.2 The etalon cavity

An *etalon*, in this project's usage, is a start vertex `r` and its
face antipode treated as a pair of reflectors. The face is the
subcube spanned by the low `subcube_dim` bits: `M = 2^subcube_dim`
vertices. The high `dim - subcube_dim` bits are pinned by `r`.

### 3.3 The transit

`ExciteCube(input_field)` does the following, exactly:

1. **Scale once, in place**: every element of the caller's buffer is
   multiplied by `input_scaling`. (This is why the Cascade host always
   copies before calling — the caller's original field is never
   touched by the public API.)
2. **For every start vertex `r = 0 … N-1`:**
   a. Reload: `memcpy` the scaled input into the working state.
   b. Forward walk: for `v = 0 … M-1`, update site `v ^ r`.
   c. Reverse walk: for `v = M-2 … 0`, update site `v ^ r`
      (skipping `v = M-1`, which was just written).
   d. The value standing at `state[r]` after the reverse walk — the
      *second* visit to `r` — is the output sample: `output[r]`.
3. Return the length-`N` output field.

A site update is a **full-star neighbor sum**:

```
state[vv] = tanh( Σ over j of state[vv ^ (1<<j)] · w[vv][j] ),   j = 0 … dim-1
```

There is no self-tap: a site's old value does not enter its own
update. The update is sequential (Gauss–Seidel style): each write is
visible to subsequent reads within the same walk. That is what gives
the transit its wave character — the disturbance propagates through
the face to the antipode and reflects back.

Two structural facts worth stating precisely:

- **Off-face taps read the initial condition.** The star sum ranges
  over all `dim` neighbors, but the walk only ever updates the `M`
  face vertices. Neighbors reached by flipping one of the pinned high
  bits are never written during the walk, so they still hold the
  scaled input. Every site update therefore mixes propagating wave
  state (on-face) with unmodified input (off-face).
- **Each start is independent.** The state is reloaded from the same
  scaled input for every `r`, so `output[r]` is a deterministic
  function of the (scaled) input field and the frozen weights. The
  output field has the same length `N` and the same vertex indexing as
  the input.

### 3.4 Cost

Per transit: `N` starts × `(2M - 1)` site updates × `dim`
multiply-adds, plus one tanh per update, plus an `N`-float reload per
start (`N²` floats of memcpy traffic per transit). Weight storage is
`N · dim` floats.

---

## 4. Stage 2: the Reservoir (a synthetic orbit)

Source: `Reservoir.h` / `Reservoir.cpp`. The header describes it as an
HypercubeESN-derived recurrent core: cube topology, delay line, input
gather, spectral-radius rescale — with **no** external-feedback path.
This is the encoder mechanism of the sibling project HypercubeWTF.

### 4.1 Frozen state

- `N = 2^dim` neurons (`dim` valid 5–16 standalone; 5–12 under the
  Cascade host).
- A **delay line** of depth `M = history_depth` (valid 1–64): the last
  `M` output fields, newest first, held in a pointer ring so aging is
  pointer rotation, not copying.
- **Input weights**: `N · dim` floats, `U(-1, 1) × input_scaling / √dim`.
- **Recurrent weights**: `N · M · dim` floats,
  `U(-1, 1) × 1 / √(dim · M)`, then rescaled to a target spectral
  radius (§4.2).
- **Bias**: `N` floats, `U(-1, 1) × bias_scaling` (0 disables).

All draws come from named substreams of one master seed:
`mix64(seed ^ 0x100000001B3 · role)` with roles Recurrent = 1,
Input = 2, Bias = 4, SrProbe = 5 (role 3 is reserved — it was
external feedback in the ESN ancestor and is deliberately never
reused, so weight draws remain comparable across the family).

### 4.2 Spectral-radius calibration

The recurrent block is rescaled so that the spectral radius of its
linearization hits `cfg.spectral_radius` (target; the realized value
is exposed via `GetRealizedSpectralRadius()`).

The estimator is a power iteration on the **companion operator** of
the delay system, acting on the full `M·N`-dimensional delay state:
the new age-0 slice is the recurrent gather over all `M` slices at
neighbor positions (no tanh — this measures the linear part), and the
older slices shift down by one. Growth is accumulated as a geometric
mean of step norms (sum of logs), with a 32-iteration burn-in,
convergence checked every 50 accumulated samples at 1e-4 relative
tolerance, and a 1500-iteration cap.

The rescale itself is a secant iteration on
`scale → realized_sr(scale) - target`: first guess `target/pre_sr`,
subsequent steps clamped to `[0.25, 4] ×` the previous scale, stopped
at 0.1 % relative error or 20 iterations.

### 4.3 The update law

Per step, synchronously for every vertex `v` (all reads see the
previous step's slices; writes go to a separate live-state buffer):

```
s(v) = Σ over i of input[v ^ (1<<i)]  · W_in[v][i]                 (staged drive)
     + Σ over age j = 0 … M-1, i of slice_j[v ^ (1<<i)] · W_rec[v][j][i]
activation(v) = tanh(s(v)) + bias[v]
new_state(v)  = (1 - leak_rate) · old_output(v) + leak_rate · activation(v)
```

Details that matter:

- The **input gather is also a neighbor star**: vertex `v` reads the
  staged drive at its `dim` neighbors, not at `v` itself.
- The recurrent taps span **all M history slices** at neighbor
  positions, so the recurrence has depth-`M` memory per step.
- The bias is added **after** tanh, so activations can leave
  `[-1, 1]` by up to `|bias|`.
- The only self-connection is the leak term.

After the vertex sweep: the slice ring rotates (oldest slot becomes
the new age-0), the live state is copied into age-0, and the staged
input is zeroed — an injected field is consumed by exactly one step.

### 4.4 Cost

Per step: `N · dim · (M + 1)` multiply-adds plus `N` tanh. Weight
storage: `N · dim · (M + 1)` floats plus `N` bias floats and the
`N · M` delay line.

---

## 5. The cascade coupling

Source: `Cascade.cpp`, `Cascade::MapOn` — the heart of the project.
For one input field `x` (length `N`):

```
1. Copy x into scratch.                      (caller's buffer is never written)
2. y = ExciteCube(scratch)                   (one etalon transit)
3. excited[i] = y[i] · interstage_scale      (the interstage gain)
4. reservoir.LoadInitialCondition(s0)        (frozen episode start)
5. for pass c = 0 … T-1:
       drive[v] = excited[(v ^ c) & (N-1)]   (xor registration remap)
       reservoir.InjectInputField(drive)
       reservoir.Step()
6. features[i] = reservoir.Outputs()[i] · readout_scale
```

Each element deserves its own paragraph.

**The frozen initial condition `s0`.** Every sample's reservoir episode
starts from the *same* full delay-line state: `N · M` floats drawn
once at Cascade construction from `U(-0.5, 0.5)` seeded by
`mix64(ic_seed ^ 0x5343000000000001)`. `LoadInitialCondition` re-homes
the slice ring and loads all `M` ages (age-0 also becomes the live
state). Consequence: **episodes are independent and the whole pipeline
is a pure deterministic map** from field to features. No state crosses
sample boundaries. This is reservoir computing machinery used as a
stateless nonlinear transform, not as a temporal memory.

**The interstage gain.** A single scalar (`interstage_scale`, finite
and > 0, default 1) between the transit output and the reservoir
drive. Because the Exciter's tanh output is bounded and the
Reservoir's input weights carry their own small scale
(`input_scaling / √dim`), this knob sets how hard the orbit is driven.
The tuned examples use markedly different values: 5.5 (synth), 1.0
(MNIST), 5.0 (Raman).

**The xor registration remap.** On pass `c`, vertex `v` is driven with
the transit sample from address `v ^ c`. XOR-translation by a constant
is an automorphism of the hypercube — a relabeling that preserves all
adjacency — so, in the root README's words, *"geometry stays put; the
registration of the field moves."* Over `T` passes the reservoir sees
the same transit field presented at `T` different registrations
(all distinct while `T ≤ N`; the mask `& (N-1)` makes the remap
well-defined regardless). Pass 0 is the identity registration. Since
`c` increments by 1, low address bits toggle fastest across passes.

**The pass count `T`.** `T ≥ 1` reservoir steps per sample (defaults
100; tuned examples use 50–60). After the final pass, the **live age-0
output** is the feature field — the readout never sees the older delay
slices.

**The readout gain.** A second scalar (`readout_scale`, default 1)
between the reservoir's live output and the Readout's input.

The dataflow discipline around this map: caller buffers are never
mutated (everything copies into per-worker scratch first), and the
serial paths publish the intermediate fields — `LastExciter` (transit
output), `LastInterstage` (drive field), `LastReservoir` (live output
before gain), `LastFeatures` (what the readout sees) — so the demos
can print per-stage mean-|value| probes ("~1 is a live field, ~0 is
crushed").

---

## 6. Stage 3: the Readout (the only trained component)

Source: `Readout.h` / `Readout.cpp` — a PIMPL façade over the vendored
HypercubeCNN library (release 1.0.3, `third_party/HypercubeCNN/`,
`HCNN.h` never appears in a public Cascade header).

### 6.1 Architecture

```
field[N] ──▶ Embed ──▶ [ Conv (+ antipodal Pool) ] × L ──▶ Flatten ──▶ Linear ──▶ outputs
```

- `dim` valid 3–30 (HypercubeCNN's range); under Cascade it equals the
  shared cube dim, so features-per-sample is the same `N`.
- `num_layers` default 1; 0 means auto: `max(1, min(dim - 2, 2))`.
  With pooling on, the stack must satisfy `layers ≤ dim - 2`.
- The optional pool after each conv is **antipodal**: it pairs each
  vertex with its bitwise complement, dropping one cube dimension per
  pool. The header warns this mixes *every* bit — including block-index
  bits of block-structured inputs — and `use_pooling = false` keeps
  vertex structure intact into the flatten head at the cost of twice
  the flattened features.
- Channels: `conv_channels` base, multiplied by `channel_growth` per
  stage. Per-conv batch-norm is available, default off.
- Task is fixed at construction: Regression or Classification
  (`num_outputs` targets or classes).

### 6.2 Training

Batch training (`Train`) is a fixed loop: Adam (default) or SGD with
cosine annealing from `lr_max` down to `lr_max · lr_min_frac` over
`lr_decay_epochs` (0 = `epochs`). By default
(`restore_best_epoch = true`) each epoch is scored — regression by
MSE, classification by accuracy — and the best weights seen are
restored at the end; an optional holdout fraction (clamped to
[0, 0.5], shuffled once by the readout seed) makes that selection
out-of-sample. Streaming `TrainStep*` variants exist as thin forwards
for hosts that run their own loop; the headers explicitly forbid
growing new training-loop policy on this façade.

### 6.3 What the minimal readout looks like

The README's aim — one layer, one channel, no pooling — is not
hypothetical: it is exactly the Raman configuration
(`conv_channels = 1`, `channel_growth = 1`, `num_layers = 1`,
`use_pooling = false`, no activation). At that size the trained model
is a single `N`-input convolution over the cube plus a linear flatten
head; everything task-relevant must already be present in the
preprocessed field.

### 6.4 Serialization

Only the Readout serializes: `SaveHcnnModel` writes a versioned HCNW
weight file plus a `.arch.json` sidecar
(format token `hypercube_cascade_readout_arch`, version 1) that
`LoadHcnnModel` validates field-by-field against the live
architecture. The preprocessors are **not** serialized — by design
they reconstruct exactly from `CascadeConfig` scalars and seeds. A
complete saved model is therefore: the config + the readout
checkpoint.

---

## 7. Determinism and the frozen/trained split

Everything upstream of the Readout is deterministic given the config:

| Component | Randomness | Seed path |
|---|---|---|
| Exciter weights | one `mt19937_64` stream | `exciter.seed` |
| Reservoir weights/bias/SR probe | named substreams | `mix64(reservoir.seed ^ role·k)` |
| Episode start `s0` | one stream | `mix64(ic_seed ^ 0x5343000000000001)` |
| Readout init | HypercubeCNN weight seed | `readout.seed` |
| Holdout shuffle | one stream | `readout.seed ^ 0x484F4C444F555400` |

The map from field to features is bit-reproducible across runs and
across worker threads (workers clone stages from the same config, so
they hold identical frozen weights). The only nondeterminism budget in
the system is whatever HypercubeCNN's threaded training introduces,
and the demo binaries are built `-O3 -march=x86-64-v2 -ffast-math`
(Release), which the scripts README explicitly warns is the only build
whose numbers should be quoted.

A related design stance from the Raman write-up: the Exciter seed is
**not a tuning parameter**. On the etalon-only sibling (LCO full
split), three independent Exciter seeds gave denormalized validation
RMSE 6.155 / 6.147 / 6.184 — a 0.037 spread on a 6.162 mean (~0.6 %).
That survey is an Etalon result, not a Cascade result, but Cascade
deliberately keeps the same seed as a frozen first-stage constant.

---

## 8. Batch machinery

Source: `Cascade.cpp` (`CollectPool`, `MapFeaturesParallel`).

The lifecycle is collect → train → predict. `CollectBatch` /
`Accuracy` / `R2` fan independent per-sample maps across a persistent
fork-join pool: plain `std::thread` workers plus mutex/condition
variables (no OpenMP), where the calling thread is worker 0 and each
worker owns a full clone of the Exciter and Reservoir plus scratch
buffers (`field`, `transit`, `excited`, `drive`). The pool and the
worker clones persist for the Cascade lifetime, growing to the
high-water mark and never shrinking. Thread count: 0 = auto
(`hw - 1`, or `hw - 2` when `hw ≥ 8`), 1 = serial, K = K workers,
always capped by the sample count. Exceptions thrown inside a worker
are captured and rethrown on the caller; a throw during collection
rolls the collected set back to its prior state. Single-sample paths
(`Run`, `Collect`, `Predict`, `PredictClass`) are always serial on the
primary stages.

The collected set is stored sample-major (`num_collected × N`
features, plus labels or targets); `TrainOnCollected` hands it to the
Readout in one call and does not clear it.

---

## 9. Cost accounting for the shipped configurations

Multiply-add counts per sample, from the formulas in §3.4/§4.4
(tanh evaluations and memory traffic excluded; per-transit reload
traffic is `N²` floats):

| Config | dim / N | M_walk | T | M_hist | Transit MACs | Orbit MACs | Orbit/Transit |
|---|---|---|---|---|---:|---:|---:|
| `cascade_synth` | 7 / 128 | 32 | 50 | 4 | 56,448 | 224,000 | 4.0× |
| `cascade_mnist` | 10 / 1024 | 32 | 50 | 2 | 645,120 | 1,536,000 | 2.4× |
| `cascade_raman` | 11 / 2048 | 32 | 60 | 8 | 1,419,264 | 12,165,120 | 8.6× |

(Transit = `N·(2·M_walk − 1)·dim`; orbit = `T·N·dim·(M_hist + 1)`.)

Frozen-weight storage is modest: Exciter `N·dim` floats; Reservoir
`N·dim·(M_hist+1)` floats. For the Raman config that is ~90 KB and
~810 KB respectively. The preprocessors are compute-heavier than they
are memory-heavy, and both scale linearly in every knob except the
per-transit `N²` reload traffic.

---

## 10. Empirical status

Stated exactly as the repository records it.

### 10.1 Synthetic gate (`cascade_synth`)

Six synthetic classes on a dim-7 cube (multi-tone carriers in the low
half, sparse signed peaks in the high half, deterministic noise;
held-out test draws are new reps of the same classes). The program
fails its own run if Cascade test accuracy is below 0.70. This is a
fast regression gate, not a benchmark claim.

### 10.2 MNIST white-noise study (`examples/mnist/WhiteNoiseFilter.md`)

Protocol: all paths pack identically (PadLowCenter, dim 10), train on
the clean 60,000-image set, then score the 10,000-image test set with
i.i.d. Gaussian noise of strength σ added to every vertex of the
packed field, no clipping. Bypass and transit columns come from the
sibling `etalon_mnist` (readout: 1 layer, 16 channels, max pool, no
activation; 20 epochs bypass, 100 transit); the cascade column is
`cascade_mnist` (same head shape; 35 epochs; T = 50, history 2,
SR 0.9 realized 0.900, leak 0.9, interstage 1.0).

| σ | bypass | transit | cascade | cascade − bypass |
|--:|------:|-------:|-------:|-----------------:|
| 0.0 | 0.979 | 0.976 | 0.975 | −0.004 |
| 0.2 | 0.967 | 0.967 | 0.967 | +0.000 |
| 0.3 | 0.934 | 0.947 | 0.954 | +0.020 |
| 0.5 | 0.799 | 0.851 | 0.887 | +0.088 |
| 0.7 | 0.648 | 0.717 | 0.779 | +0.131 |
| 1.0 | 0.469 | 0.535 | 0.600 | +0.131 |

(Full table with all eleven σ points and the transit−bypass column is
in the write-up.)

The reading the write-up gives: on clean or near-clean fields the
cascade is a near-unity passthrough (−0.004 at σ = 0); from σ = 0.3
upward the cascade leads the single-stage transit, and both lead
bypass — i.e., **the second stage adds filtering on top of what the
first stage already adds**, and the ordering is monotone in the noise
level. The write-up is explicit that this configuration is kept small
for fast experimentation and is not chasing a record MNIST score.

### 10.3 Raman baseline extraction

Cascade result on the full LCOHard split (10,000 train / 2,000
validation), with the readout at the minimum the project is aiming
for — one conv layer, one channel, no pooling: denormalized RMSE
**4.779 training / 4.819 validation** (raw counts; per-spectrum
input-anchored normalization, scored against raw labels). The
0.04-count train/validation gap shows no overfit at this readout
capacity. The fit ran 60 epochs with cosine decay to a 4 % floor;
best epoch was 59 of 60 and the last ten epochs moved 0.07 counts in
total, so the curve had flattened into its floor. Details in
`examples/RamanBaselineExtraction/README.md`.

The single-stage comparison arm is the cleanest available control:
the HypercubeEtalon sibling now runs the identical Exciter (same
seed, subcube, input and weight scales) with the reservoir removed,
on the same split, the same head shape, and the same 60-epoch budget.
It scored **4.705 training / 4.767 validation** — 0.05 counts (about
1 %) ahead of the cascade on validation. The held-out overlays of the
two hosts are indistinguishable; the gap is in the score, not in the
extract. The cascade therefore matches the etalon alone on this
nearly clean data while carrying the second stage that §10.2 found to
be a filter under rising noise. Whether that margin opens on noisier
spectra has not been measured.

### 10.4 What has not been shown

Also recorded in the repo's own TODOs: the HypercubeWTF
(reservoir-only) arm is not yet in the MNIST comparison, and the
reversed order (Reservoir → Etalon → Readout) is untested. So the full
four-way claim — cascade beats bypass, etalon-alone, *and*
reservoir-alone — is at present demonstrated only against bypass and
etalon-alone, on one dataset, under one noise protocol. On Raman
LCOHard the etalon-alone arm has been run (§10.3) and sits at parity
with the cascade — 0.05 counts ahead in score, indistinguishable in
the extracts; no bypass, reservoir-only, or added-noise arm has been
run there.

---

## 11. Relation to reservoir computing

The Reservoir stage is orthodox echo-state machinery: frozen random
recurrent weights, spectral-radius calibration of the linearized
dynamics, leaky-integrator update, a delay line. The departures are
what make the cascade what it is:

1. **No temporal task.** Classical reservoir computing exploits the
   reservoir's fading memory over an input *sequence*. Here the input
   is a static field; the "sequence" is synthesized by presenting that
   one field at `T` xor-shifted registrations from a fixed initial
   condition. The reservoir is used as a spatial mixer with dynamics,
   not as a memory.
2. **Frozen per-sample episodes.** Reloading the same `s0` for every
   sample removes all cross-sample state and makes the transform a
   pure function — trainable-pipeline semantics, unusual for reservoir
   systems.
3. **A structured, not random, topology.** Both stages couple only
   hypercube neighbors. The recurrence is random in its *weights* but
   exactly regular in its *graph* — every neuron has degree `dim` per
   delay slice, and the readout CNN convolves over the same graph.
4. **A second, non-reservoir stage in front.** The Exciter is not a
   small reservoir: no leak, no delay, no orbit. It is a deterministic
   sequential wave map whose output samples are point-values of
   reflected transits. The cascade thesis is precisely that these two
   different frozen nonlinearities compose productively.
5. **The readout is a CNN, not a linear regression.** Standard ESN
   practice trains a linear readout; here the head is a (deliberately
   tiny) convolutional network on the same cube, and the project goal
   is to shrink it toward the linear limit rather than start there.

---

## 12. Limitations and open questions

Honest edges of the current system, all visible in the source:

- **Scale ceiling.** The shared dim is capped at 12 (`N = 4096`) by
  the Exciter's cost cap; the per-transit `N²` reload traffic is the
  reason the cap exists.
- **Two gains, hand-tuned.** `interstage_scale` and `readout_scale`
  are per-task scalar knobs (5.5 / 1.0 / 4.0 across the three
  examples). Nothing in the system sets them automatically; the demos'
  stage-scale probes exist to tune them by eye.
- **Evidence base is narrow** (§10.4): one packed-image dataset with
  one noise protocol, one signal-regression task without a recorded
  number yet, and a synthetic gate.
- **`T` and the remap schedule are fixed policy.** Passes always remap
  by consecutive integers `0 … T-1`; alternative registration
  schedules are unexplored in this codebase.
- **The readout sees only age-0.** The delay line exists for the
  dynamics, but the feature field is the final live slice; older
  slices are discarded at readout time.

---

## 13. Source map

| Concern | File |
|---|---|
| Host config (dim, T, gains, ic_seed, threads) | `CascadeConfig.h` |
| Pipeline, episode loop, xor remap, thread pool | `Cascade.h` / `Cascade.cpp` |
| Etalon transit stage | `Exciter.h` / `Exciter.cpp` |
| Synthetic-orbit stage | `Reservoir.h` / `Reservoir.cpp` |
| Trained head (HCNN façade, HCNW I/O) | `Readout.h` / `Readout.cpp` |
| Vendored CNN core (release 1.0.3) | `third_party/HypercubeCNN/` |
| MNIST noise study | `examples/mnist/WhiteNoiseFilter.md`, `noise.md` |
| Raman task write-up | `examples/RamanBaselineExtraction/README.md` |
| Sibling single-stage hosts | `HypercubeEtalon`, `HypercubeWTF` (separate repos) |
