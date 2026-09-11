// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 David Liptak

#pragma once

/**
 * @file HypercubeCNN.h
 * @brief Umbrella include for the full public SDK surface.
 *
 * HypercubeCNN is a dependency-free C++23 hypercube CNN core for research and
 * systems integration.  The product is the core (`HCNN`); this umbrella also
 * pulls optional helpers.  Examples and helpers are not the reason the library
 * exists — use them when useful, skip them when not.
 *
 * Pulls in:
 *   - `HCNN.h` / `HCNNInput.h` — core front door + full-capacity input types
 *   - `HCNNArch.h`          — LayerSpec, apply_arch, HCNNConfig::Build
 *   - `HCNNTrainHelpers.h`  — optional metrics, cosine LR, checkpoints, flat dataset
 *   - `HCNNSpatialAug.h`    — optional 2D augmentation
 *   - `HCNNSpatialEmbed.h`  — optional 2D → length-N packing
 *
 * Prefer this single include when the host wants arch helpers, train helpers,
 * and/or spatial preprocess.  Minimal integrations that only need the graph
 * can `#include "HCNN.h"` alone.
 *
 * Private implementation headers (`HCNNNetwork`, layers, `ThreadPool`) are
 * **not** included and **not** installed — apps never need them.
 */

#include "HCNN.h"
#include "HCNNArch.h"
#include "HCNNTrainHelpers.h"
#include "HCNNSpatialAug.h"
#include "HCNNSpatialEmbed.h"
