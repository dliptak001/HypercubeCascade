# The Cascade as a white-noise filter

HypercubeWTF asked a simple question. Pack an image onto the cube, dump
white noise onto that field, and ask whether a hypercube reservoir
helps the CNN head more than handing the noisy pack straight to it.
Their answer was yes. On clean digits the reservoir costs almost
nothing. Under heavy noise it holds eight or nine points that the
pack-only path loses. The write-up is
`HypercubeWTF/examples/mnist/WhiteNoiseFilter.md`.

Etalon ran the same protocol with one etalon transit instead of a
reservoir episode. Same packing, same kind of head, same noise on the
packed field. The answer was yes again.

That same experiment is run on Cascade. One etalon transit, then a
reservoir orbit, then the same kind of head.

On a clean test set the three paths are still almost tied — bypass
0.979, etalon transit 0.976, cascade 0.975. At σ = 0.2 they sit on the
same number, 0.967. From σ = 0.3 on, cascade is ahead of the etalon
transit, and both stay ahead of bypass. At σ = 0.5 cascade is 0.887
against transit 0.851 and bypass 0.799 — 3.6 points over the transit,
8.8 over bypass. At σ = 1.0 those gaps are 6.5 and 13.1 (0.600 vs
0.535 vs 0.469).

MNIST is just a convenient test bed. For this example the cascade
configuration is not scaled to reach test accuracies ≥ 99.5%, though
they could be. Instead, the architecture is kept small and fast,
allowing rapid experimentation for a study relative to the bypass
configuration.

---

## How the comparison works

Every image is packed the same way: the full 28×28 digit in the low
addresses, a centered crop filling the rest of the 1024-vertex cube.
Each path trains on the clean 60,000-image training set. At test time
we add independent Gaussian noise of strength σ to every vertex of
the packed field — no clipping — and score the 10,000-image test set.

What the readout sees:

- **Bypass** — the packed field, noise and all.
- **Etalon transit** — that same field after one frozen transit.
- **Cascade** — that same field after one frozen transit and a
  reservoir orbit.

---

## What happened

![MNIST test noise: cascade vs etalon transit vs Bypass](cascade_mnist_noise_comp.png)

Bypass has no trouble fitting the clean training set. It just does not
recognize those digits once the test field is full of snow.

---

## The numbers

| sigma | bypass | transit | cascade | transit − bypass | cascade − bypass |
|------:|-------:|--------:|--------:|-----------------:|-----------------:|
| 0.0 | 0.979 | 0.976 | 0.975 | −0.003 | −0.004 |
| 0.1 | 0.977 | 0.974 | 0.972 | −0.003 | −0.005 |
| 0.2 | 0.967 | 0.967 | 0.967 | +0.000 | +0.000 |
| 0.3 | 0.934 | 0.947 | 0.954 | +0.014 | +0.020 |
| 0.4 | 0.875 | 0.910 | 0.928 | +0.035 | +0.053 |
| 0.5 | 0.799 | 0.851 | 0.887 | +0.052 | +0.088 |
| 0.6 | 0.723 | 0.784 | 0.839 | +0.062 | +0.116 |
| 0.7 | 0.648 | 0.717 | 0.779 | +0.069 | +0.131 |
| 0.8 | 0.580 | 0.649 | 0.712 | +0.069 | +0.132 |
| 0.9 | 0.521 | 0.587 | 0.656 | +0.066 | +0.135 |
| 1.0 | 0.469 | 0.535 | 0.600 | +0.066 | +0.131 |

---

## Settings

What `etalon_mnist.cpp` used for the bypass and transit columns.

| | |
|--|--|
| Cube | dim 10, 1024 vertices |
| Etalon transit | `subcube_dim = 5` (32 steps) |
| Input / weight scale | 1.0 / 0.15 |
| Weight seed | 3458567978345987 |
| Readout | 1 layer, 16 channels, max pool, no activation |
| Epochs | 100 transit, 20 bypass |

What `cascade_mnist.cpp` used for the cascade column.

| | |
|--|--|
| Cube | dim 10, 1024 vertices |
| Exciter | `subcube_dim = 5` (32 steps), input / weight scale 0.1 / 0.15 |
| Reservoir | T = 50, history depth 2, spectral radius 0.9 (realized 0.900), leak 0.9, input scale 1.0, bias 0 |
| Interstage / readout scale | 1.0 / 1.0 |
| Seeds | exciter 3458567978345987, reservoir 13871537636959942979, ic_seed 12 |
| Readout | 1 layer, 16 channels, max pool, no activation |
| Epochs | 35 |


---

## Todo
1. Include HypercubeWTF data in the comparison.
2. Reverse the preprocessor order:  Reservoir -> Etalon -> Readout
