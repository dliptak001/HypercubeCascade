# Test-field white noise — mental model

How to picture **i.i.d. Gaussian test noise** on the packed HypercubeWTF field
(MNIST demo / campaigns). For the reservoir-as-pre-filter study, see
[WhiteNoiseFilter.md](WhiteNoiseFilter.md).

## Protocol (what σ actually is)

| | |
|--|--|
| Where | After pack, on the length-N field, before `PredictClass` |
| Distribution | i.i.d. N(0, σ) per vertex |
| Clamp | **None** — values may leave [-1, 1] |
| Train | Stays clean |
| Not this | Train 28×28 aug |

Single-shot knob: `kTestNoiseSigma` in `wtf_mnist.cpp`.  
Campaign sweep: `MakeTestNoiseSigmaGrid()` / `RunTestNoiseSigmaCampaign`.

## Clean field picture

Packed MNIST (demo defaults):

- Field lives in **[-1, 1]**
- Background / pad ≈ **−1** (black)
- Ink ≈ **+1** (white)
- **Ink–background gap ≈ 2** — that is the useful contrast
- Noise hits **every** vertex (image verts and pad verts)

So σ is “how far a typical vertex jitters” on that scale. Compare σ to the
**gap of 2**, not to an abstract radio S/N.

## Corruption ladder (the table)

| σ | Typical jiggle (±1σ) | What the field “looks like” |
|---|----------------------|-----------------------------|
| **0.1** | ±0.1 on a range of 2 | Light film grain. Digit shape obvious; pad still clearly “black.” Mild stress. |
| **0.3** | ±0.3 | Heavy grain / static. Edges soft; a 1–2σ hit (±0.3–0.6) is a real bite out of contrast. Stress starts to show in accuracy. |
| **0.5** | ±0.5 | Strong snow. Noise std = 1/4 of full scale (half of peak \|s\|). 2σ spikes (±1) can flip a black pad vertex toward mid-grey or punch a hole in ink. Digit often still “there” to a human, but pack→CNN is badly mismatched. WhiteNoiseFilter headline case (bypass ~0.85 vs reservoir ~0.93). |
| **0.75** | ±0.75 | Blizzard. Typical noise is a large fraction of the ink–bg gap; many vertices leave [-1, 1]. Structure is half-guess. |
| **1.0** | ±1.0 | Noise std = peak \|s\|. A typical draw is as large as “full white or full black.” Field is mostly noise with a weak digit ghost. |
| **≥1.5** | ±≥1.5 | Near white-out → chance (~0.10). Little left to pre-filter productively. |

### How far to sweep

- **Default study max: 0.5** (headline A/B lives here).
- **Optional stretch: 0.75–1.0** (post-stress cliff / ghost digit).
- **Avoid past ~1.0–1.5** for normal campaigns — both arms die toward chance.
