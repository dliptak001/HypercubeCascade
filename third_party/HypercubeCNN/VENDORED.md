# Vendored HypercubeCNN

This directory is a **copy of the HypercubeCNN library sources** used by
HypercubeCascade. It is not a git submodule. Upstream is the public HypercubeCNN
project.

## What is in this folder

| Item | Role |
|------|------|
| `HCNN*.h` / `HCNN*.cpp`, `HypercubeCNN.h`, `ThreadPool.h` | Library sources (the product) |
| `LICENSE` | Apache-2.0 from upstream |
| `CMakeLists.txt` | **HypercubeCascade only** — how this tree is built *inside* Cascade. Not the upstream project CMake. |
| `VENDORED.md` | This note |

No examples, Python package, tests, or docs from HypercubeCNN — only the core
needed to link HypercubeCascade.

## Version pin

| | |
|--|--|
| **Upstream project** | HypercubeCNN |
| **Release** | **1.0.3** |
| **Git commit** | `a60b6896e929eaae2429615692f5ce50109b3215` |

Library sources in this directory match that release (re-verify with a full
re-copy when bumping).

## How to update

1. Check out HypercubeCNN at the commit you want.
2. Copy the library sources into this directory (same set of files as above).
3. Leave this directory’s `CMakeLists.txt` alone unless the Cascade build needs a change.
4. Update the **Version pin** table in this file.

Do not edit vendored library sources in place for long-lived fixes — fix
upstream, then re-copy.
