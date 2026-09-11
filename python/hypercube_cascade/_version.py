# Single source of truth for the package version.
# - pyproject.toml reads this via scikit-build-core dynamic metadata
# - hypercube_cascade.__version__ imports it
# - bindings.cpp gets the same string at compile time via CMake
__version__ = "1.0.4"
