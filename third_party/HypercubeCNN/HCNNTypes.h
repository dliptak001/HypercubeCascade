// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 David Liptak

#pragma once

/**
 * @file HCNNTypes.h
 * @brief Public enums for the HypercubeCNN SDK front door.
 *
 * Included by `HCNN.h`.  Layer/orchestrator headers are private (not installed).
 */

#include <cstdint>

namespace hcnn {

/// Activation after convolution (and optional batch-norm).
///
/// - `NONE`: identity
/// - `RELU` / `LEAKY_RELU`: rectified linear (He init when c_in > 1)
/// - `TANH`: smooth, bounded (-1, 1) (Xavier init)
enum class Activation { NONE, RELU, LEAKY_RELU, TANH };

/// Weight-update rule (set via `HCNN::SetOptimizer`).  Default is Adam.
enum class OptimizerType { SGD, ADAM };

/// Antipodal pool reduction.
enum class PoolType { MAX, AVG };

/// Task the network is trained for.  Selects train API family and loss.
///
/// - `Classification`: int class targets; softmax + cross-entropy
/// - `Regression`: float targets of length `num_outputs`; MSE
///
/// Loss is fixed by task (no separate loss enum on the public API).
enum class TaskType { Classification, Regression };

} // namespace hcnn
