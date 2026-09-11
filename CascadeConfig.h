#pragma once

#include "Exciter.h"
#include "Readout.h"
#include "Reservoir.h"

#include <cstddef>
#include <cstdint>

/// Construction-time knobs for @ref Cascade.
///
/// Set @c dim once. Construction stamps it onto @c exciter.dim,
/// @c reservoir.dim, and @c readout.dim. Nested @c dim fields are not
/// independent knobs. @c exciter.subcube_dim must stay in **[1, dim]**
/// (default 6 is legal only when @c dim >= 6).
///
/// Typical host knobs: @c dim, @c exciter scales / seed / subcube_dim,
/// @c reservoir scales / seed / history_depth, @c readout.num_outputs /
/// @c readout.task / training hyperparameters, @c T, @c interstage_scale,
/// @c readout_scale.
struct CascadeConfig
{
    /// Cube dim for Exciter, Reservoir, and Readout. N = 2^dim.
    /// Valid range **[5, 12]**.
    size_t dim = 8;

    ExciterConfig exciter{};
    ReservoirConfig reservoir{};
    ReadoutConfig readout{};

    /// Parallel workers for bulk @ref Cascade::CollectBatch /
    /// @ref Cascade::Accuracy / @ref Cascade::R2. 0 = auto (leave 1–2
    /// cores free), 1 = serial, K = K workers. Single-sample
    /// @ref Cascade::Collect / @ref Cascade::Run is always serial.
    ///
    /// Auto policy: max(1, hw − 1), or max(1, hw − 2) when hw ≥ 8.
    size_t collect_threads = 0;

    /// Reservoir drive-pass count per sample. Must be ≥ 1.
    size_t T = 100;

    /// Gain on the Exciter output before the reservoir orbit.
    /// Must be finite and > 0. Default 1 (passthrough).
    float interstage_scale = 1.0f;

    /// Gain on the Reservoir output before the Readout.
    /// Must be finite and > 0. Default 1 (passthrough).
    float readout_scale = 1.0f;

    /// Seed for the frozen episode start s0 (N × M). Not a weight seed.
    uint64_t ic_seed = 1;
};
