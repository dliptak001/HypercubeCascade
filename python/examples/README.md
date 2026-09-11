# Python examples

Demo scripts for the `hypercube-cascade` API. They call only the installed
package.

| Script | What it shows |
|--------|----------------|
| [`synthetic_classification.py`](synthetic_classification.py) | Multi-class length-N fields → `fit` → train / test accuracy |

## How to run

These files live in the GitHub repository. They are not part of `pip install`
— use the [Quick start](../README.md#quick-start) if you only want a snippet.

From a clone of HypercubeCascade (repository root), after installing the
package:

```bash
pip install hypercube-cascade
# or from this tree:  pip install .
python python/examples/synthetic_classification.py
```

## What these are not

- **Not** the C++ demos (`cascade_synth`, `cascade_mnist`, `cascade_raman`) or
  the study write-ups. Those live under [`examples/`](../../examples/).
- **Not** automated tests. Package tests are [`tests/test_basic.py`](../tests/test_basic.py).
- **Not** hard tasks. Synthetic multi-class fields are easy onboarding so the
  map API is obvious; do not cite their metrics as research results.

## Going further

| Want… | See… |
|-------|------|
| Full Python API | [`docs/Python_SDK.md`](../../docs/Python_SDK.md) |
| C++ product guide | [`docs/CPP_SDK.md`](../../docs/CPP_SDK.md) |
| Package readme | [`README.md`](../README.md) |
