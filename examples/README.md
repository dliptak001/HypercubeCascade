# HypercubeCascade examples

Four runnable programs. Each builds a length-N field, sends it through
the Cascade (Exciter then Reservoir then Readout), and scores a
held-out set.

Cascade itself is collect, train, predict. These programs own
everything outside that: synthetic patterns, MNIST IDX, and Raman
spectra on disk.

| Program | What it is for | Data files? |
|---------|----------------|-------------|
| `cascade_synth` | Fast gate | No |
| `cascade_mnist` | Packed handwritten digits | Yes — see below |
| `cascade_raman` | Raman baseline regression | Yes — not shipped |
| `cascade_raman_extract` | Write selected baseline extracts | Yes — needs a trained stem |

Knobs live at the top of each program: `MakeBaseConfig()` and the
demo-only constexprs beside it. Raman shares `MakeBaseConfig()` from
`RamanBaselineExtraction/BaselineExtractor.h`.

---

## `cascade_synth`

The program invents six classes on a small cube (`N = 128`): two
tones in the low addresses, a handful of signed spikes in the high
addresses, and a little noise. The test set is new draws of those
same classes, not the training fields again.

A run fails if Cascade test accuracy is below 0.70.

---

## `cascade_mnist`

Ten-class handwritten digits. A 28×28 image is 784 pixels; the cube
here is `N = 1024` (dim 10). HypercubeCNN's spatial embed lays the
full digit into the low addresses and a centered crop into the
leftover budget. That packed field is what the Cascade sees.

This example is not chasing a record MNIST score.

It can train on a clean pack and then score a test-only white-noise
ladder. The write-up is
[`mnist/WhiteNoiseFilter.md`](mnist/WhiteNoiseFilter.md).

### Data setup

`cascade_mnist` does not read MNIST from this git clone. It looks only
at:

```text
C:\HypercubeCascade\data\
```

Put the four uncompressed IDX files there (see
[Appendix: MNIST files](#appendix-mnist-files)). The dataset is not in
this repository.

---

## `cascade_raman`

A Raman spectrum is a line of 2048 amplitudes: sharp molecular peaks
sitting on a slow, unwanted background. This example asks the Cascade
to estimate that background. The cube is the same length as the
spectrum (`N = 2048`, dim 11), so each bin is already one address —
there is nothing to pack.

The spectra are about 1 GB and are not in the repository. The programs
look only at:

```text
C:\HypercubeCascade\RamanSpectraLCOHard\
```

`cascade_raman` trains and scores. `cascade_raman_extract` writes
selected `.pred.txt` rows. `plot_extracted.py` overlays them.

Task write-up: [`RamanBaselineExtraction/README.md`](RamanBaselineExtraction/README.md).

---

## Folder layout

```text
examples/
  README.md
  common/                 shared helpers (not the core library)
  synth/cascade_synth.cpp
  mnist/                  MNIST demo + IDX loader
  RamanBaselineExtraction/
```

---

## Appendix: MNIST files

**Location:** `C:\HypercubeCascade\data\`

**Required files** (uncompressed IDX, exact names):

```text
train-images-idx3-ubyte
train-labels-idx1-ubyte
t10k-images-idx3-ubyte
t10k-labels-idx1-ubyte
```

These are the usual public MNIST binaries (LeCun et al.). We do not
ship them in git.

**Download example** (run from `C:\HypercubeCascade\data`, or save into it):

```text
curl -L -O https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz
curl -L -O https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz
curl -L -O https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz
curl -L -O https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz
gunzip *.gz
```

On Windows, any tool that downloads those four `.gz` files and
decompresses them into `C:\HypercubeCascade\data` is fine.
