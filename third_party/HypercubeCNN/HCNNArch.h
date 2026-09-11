// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 David Liptak

#pragma once

/**
 * @file HCNNArch.h
 * @brief Architecture product surface: layer specs, param accounting, build helper.
 *
 * Optional public SDK product (installed).  Hosts can describe a stack as a
 * list of LayerSpec, apply it to an HCNN, print a summary, or build a full
 * net from HCNNConfig in one call.  Not required for core-only integrations.
 *
 * Header-only (depends on HCNN.h).  Also pulled in by HypercubeCNN.h.
 */

#include "HCNN.h"

#include <memory>
#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace hcnn {

// =============================================================================
// LayerSpec — one body step (conv or antipodal pool)
// =============================================================================

/// One stack step: Hamming conv or antipodal pool.
struct LayerSpec {
    enum class Kind { Conv, Pool };

    Kind kind = Kind::Conv;

    int c_out = 16;
    Activation activation = Activation::RELU;
    bool use_bias = true;
    bool use_bn = false;

    PoolType pool_type = PoolType::MAX;

    static LayerSpec Conv(int c_out,
                          Activation act = Activation::RELU,
                          bool bias = true,
                          bool bn = false) {
        LayerSpec L;
        L.kind = Kind::Conv;
        L.c_out = c_out;
        L.activation = act;
        L.use_bias = bias;
        L.use_bn = bn;
        return L;
    }

    static LayerSpec Pool(PoolType type = PoolType::MAX) {
        LayerSpec L;
        L.kind = Kind::Pool;
        L.pool_type = type;
        return L;
    }
};

// =============================================================================
// Param summary (matches HCNN::GetWeightCount layout)
// =============================================================================

struct ArchParamSummary {
    long long total = 0;              ///< Full GetWeightCount after RandomizeWeights
    long long readout = 0;
    long long flatten_features = 0;
    int final_dim = 0;
    int final_N = 0;
    int last_channels = 0;
    int num_conv = 0;
    int num_pool = 0;
    std::vector<long long> conv_params;  ///< Per-conv blob floats (kernel+bias+BN*)
};

[[nodiscard]] inline const char* activation_name(Activation a) {
    switch (a) {
        case Activation::NONE:       return "NONE";
        case Activation::RELU:       return "RELU";
        case Activation::LEAKY_RELU: return "LEAKY_RELU";
        case Activation::TANH:       return "TANH";
    }
    return "?";
}

[[nodiscard]] inline const char* pool_name(PoolType t) {
    switch (t) {
        case PoolType::MAX: return "MAX";
        case PoolType::AVG: return "AVG";
    }
    return "?";
}

[[nodiscard]] inline const char* optimizer_name(OptimizerType o) {
    switch (o) {
        case OptimizerType::SGD:  return "SGD";
        case OptimizerType::ADAM: return "ADAM";
    }
    return "?";
}

[[nodiscard]] inline const char* task_name(TaskType t) {
    switch (t) {
        case TaskType::Classification: return "Classification";
        case TaskType::Regression:     return "Regression";
    }
    return "?";
}

/**
 * Walk a layer list: track DIM/N/channels, per-conv blob size, FLATTEN readout.
 *
 * Total matches `HCNN::GetWeightCount` after `RandomizeWeights`:
 *   conv: kernel (c_out*c_in*K) + bias? + if BN: 4*c_out (γ, β, run_mean, run_var)
 *   readout: num_outputs * (c_final * N_final) + num_outputs
 *
 * @param start_dim  In [3, 30] (same as HCNN).
 * @throws std::runtime_error on empty stack, no conv, invalid dims/channels,
 *         or pooling past the floor (current_dim < 2).
 */
[[nodiscard]] inline ArchParamSummary summarize_arch(
    int start_dim,
    int num_outputs,
    int input_channels,
    const std::vector<LayerSpec>& layers) {
    if (start_dim < 3 || start_dim > 30)
        throw std::runtime_error(
            "summarize_arch: start_dim must be in [3, 30], got "
            + std::to_string(start_dim));
    if (num_outputs < 1)
        throw std::runtime_error("summarize_arch: num_outputs must be >= 1");
    if (input_channels < 1)
        throw std::runtime_error("summarize_arch: input_channels must be >= 1");
    if (layers.empty())
        throw std::runtime_error("summarize_arch: need at least one layer");

    ArchParamSummary s;
    int d = start_dim;
    // N = 2^d; start_dim <= 30 so this fits in signed 32-bit int.
    int N = 1 << d;
    int c_in = input_channels;
    s.last_channels = c_in;

    for (const auto& L : layers) {
        if (L.kind == LayerSpec::Kind::Conv) {
            if (L.c_out < 1)
                throw std::runtime_error("summarize_arch: conv c_out must be >= 1");
            // K = DIM + 1 (self + Hamming-1 neighbors)
            long long k_params =
                static_cast<long long>(c_in) * L.c_out * (d + 1)
                + (L.use_bias ? L.c_out : 0);
            if (L.use_bn)
                k_params += 4LL * L.c_out;  // gamma, beta, running_mean, running_var
            s.conv_params.push_back(k_params);
            s.total += k_params;
            c_in = L.c_out;
            s.last_channels = L.c_out;
            ++s.num_conv;
        } else {
            // Match HCNNNetwork::add_pool: require current_dim >= 2.
            if (d < 2)
                throw std::runtime_error(
                    "summarize_arch: cannot pool at current_dim="
                    + std::to_string(d) + " (need >= 2)");
            d -= 1;
            N = 1 << d;
            ++s.num_pool;
        }
    }

    if (s.num_conv < 1)
        throw std::runtime_error("summarize_arch: need at least one Conv layer");

    s.final_dim = d;
    s.final_N = N;
    s.flatten_features = static_cast<long long>(s.last_channels) * N;
    s.readout = s.flatten_features * num_outputs + num_outputs;
    s.total += s.readout;
    return s;
}

/// Convenience: use sizing accessors from an already-constructed (empty-body) net.
[[nodiscard]] inline ArchParamSummary summarize_arch(
    const HCNN& net,
    const std::vector<LayerSpec>& layers) {
    return summarize_arch(net.GetStartDim(), net.GetNumOutputs(),
                          net.GetInputChannels(), layers);
}

/**
 * Append layers from `layers` onto `net` (typically a freshly constructed HCNN
 * with no AddConv/AddPool yet).  Validates with summarize_arch first.
 */
inline void apply_arch(HCNN& net, const std::vector<LayerSpec>& layers) {
    (void)summarize_arch(net, layers);
    for (const auto& L : layers) {
        if (L.kind == LayerSpec::Kind::Conv)
            net.AddConv(L.c_out, L.activation, L.use_bias, L.use_bn);
        else
            net.AddPool(L.pool_type);
    }
}

/**
 * Validate + apply using explicit dims (for demos that print before construct).
 * Prefer `apply_arch(HCNN&, layers)` when the net already exists.
 */
inline void apply_arch(HCNN& net,
                       int start_dim,
                       int num_outputs,
                       int input_channels,
                       const std::vector<LayerSpec>& layers) {
    if (net.GetStartDim() != start_dim
        || net.GetNumOutputs() != num_outputs
        || net.GetInputChannels() != input_channels) {
        throw std::runtime_error(
            "apply_arch: HCNN sizing does not match (dim/outputs/channels)");
    }
    apply_arch(net, layers);
}

inline void print_arch(std::ostream& os,
                       int start_dim,
                       int num_outputs,
                       int input_channels,
                       const std::vector<LayerSpec>& layers,
                       const ArchParamSummary& sum) {
    int d = start_dim;
    int N = 1 << d;
    int c_in = input_channels;

    os << "\nArchitecture: ";
    bool first_line = true;
    for (const auto& L : layers) {
        if (!first_line)
            os << "              -> ";
        first_line = false;

        if (L.kind == LayerSpec::Kind::Conv) {
            os << "Conv(" << c_in << "->" << L.c_out
               << ", " << activation_name(L.activation);
            if (L.use_bias) os << ", bias";
            if (L.use_bn)   os << ", BN";
            os << ")  DIM=" << d << "  N=" << N << "\n";
            c_in = L.c_out;
        } else {
            os << "Pool(" << pool_name(L.pool_type) << ")  DIM "
               << d << "->" << (d - 1)
               << "  N " << N << "->" << (N / 2) << "\n";
            d -= 1;
            N = 1 << d;
        }
    }

    os << "              -> FLATTEN\n"
       << "              -> Linear(" << sum.flatten_features
       << " -> " << num_outputs << ")\n"
       << "Parameters:   " << sum.total << " (";
    for (size_t i = 0; i < sum.conv_params.size(); ++i) {
        if (i) os << " + ";
        os << sum.conv_params[i] << " conv" << (i + 1);
    }
    os << " + " << sum.readout << " readout)\n\n";
}

// =============================================================================
// HCNNConfig — knobs + one-shot Build()
// =============================================================================

/**
 * All construction knobs for a standard application net.
 *
 * Typical use:
 * @code
 *   HCNNConfig cfg;
 *   cfg.start_dim = 10;
 *   cfg.num_outputs = 10;
 *   cfg.layers = {
 *       LayerSpec::Conv(16),
 *       LayerSpec::Pool(),
 *       LayerSpec::Conv(32),
 *   };
 *   auto net = cfg.Build();  // unique_ptr<HCNN>; randomized + Adam
 * @endcode
 */
struct HCNNConfig {
    int start_dim = 10;
    int num_outputs = 10;
    int input_channels = 1;
    TaskType task = TaskType::Classification;
    size_t num_threads = 0;

    std::vector<LayerSpec> layers;

    /// Applied after RandomizeWeights when `randomize` is true.
    OptimizerType optimizer = OptimizerType::ADAM;
    float adam_beta1 = 0.9f;
    float adam_beta2 = 0.999f;
    float adam_eps = 1e-8f;

    bool randomize = true;
    float weight_scale = 0.0f;   ///< <= 0 → He/Xavier per layer
    uint64_t weight_seed = 42;   ///< Full 64-bit weight-init master seed

    /// Param summary for this config (validates layers).
    [[nodiscard]] ArchParamSummary summarize() const {
        return summarize_arch(start_dim, num_outputs, input_channels, layers);
    }

    /**
     * Construct HCNN, apply layers, optional RandomizeWeights, SetOptimizer.
     * Returns unique_ptr for exclusive ownership (HCNN is movable but not
     * copyable; unique_ptr is the usual host pattern).
     *
     * Order: construct → apply_arch → RandomizeWeights → SetOptimizer.
     * Keep the LayerSpec list (or this config) if you will save/load weights:
     * HCNW stores parameters only, not the layer graph.
     */
    [[nodiscard]] std::unique_ptr<HCNN> Build() const {
        auto sum = summarize();  // validate early
        (void)sum;

        auto net = std::make_unique<HCNN>(start_dim, num_outputs, input_channels,
                                          task, num_threads);
        apply_arch(*net, layers);

        if (randomize)
            net->RandomizeWeights(weight_scale, weight_seed);

        net->SetOptimizer(optimizer, adam_beta1, adam_beta2, adam_eps);
        return net;
    }
};

} // namespace hcnn
