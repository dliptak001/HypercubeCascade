// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 David Liptak

#include "HCNNPool.h"
#include "ThreadPool.h"
#include <cstdint>
#include <stdexcept>
#include <string>

namespace hcnn {

// Pool work per element is lighter than conv, so use a higher threshold.
static constexpr int POOL_THREAD_DIM_THRESHOLD = 14;

HCNNPool::HCNNPool(int input_dim, PoolType type)
    : input_dim(input_dim), output_dim(input_dim - 1),
      input_N(0), output_N(0), type(type) {
    // Align with HCNNNetwork: need dim >= 2 to pool; N = 2^dim in signed int.
    if (input_dim < 2 || input_dim > 30) {
        throw std::runtime_error(
            "HCNNPool requires 2 <= input_dim <= 30, got "
            + std::to_string(input_dim));
    }
    input_N  = static_cast<int>(std::uint32_t{1} << input_dim);
    output_N = static_cast<int>(std::uint32_t{1} << (input_dim - 1));
}

void HCNNPool::forward(const float* in, float* out, int num_channels,
                       std::vector<int>* max_indices) const {
    if (max_indices && type == PoolType::MAX) {
        max_indices->resize(num_channels * output_N);
    }

    const uint32_t anti_mask = (1u << input_dim) - 1;
    const bool use_threads = thread_pool && input_dim >= POOL_THREAD_DIM_THRESHOLD;

    auto do_channels = [&](int c_begin, int c_end) {
        if (type == PoolType::MAX) {
            if (max_indices) {
                for (int c = c_begin; c < c_end; ++c) {
                    const float* chan_in = in + c * input_N;
                    float* chan_out = out + c * output_N;
                    int* idx = max_indices->data() + c * output_N;
                    for (int v = 0; v < output_N; ++v) {
                        int v_anti = v ^ anti_mask;
                        if (chan_in[v] >= chan_in[v_anti]) {
                            chan_out[v] = chan_in[v];
                            idx[v] = v;
                        } else {
                            chan_out[v] = chan_in[v_anti];
                            idx[v] = v_anti;
                        }
                    }
                }
            } else {
                for (int c = c_begin; c < c_end; ++c) {
                    const float* chan_in = in + c * input_N;
                    float* chan_out = out + c * output_N;
                    for (int v = 0; v < output_N; ++v) {
                        int v_anti = v ^ anti_mask;
                        chan_out[v] = (chan_in[v] >= chan_in[v_anti])
                                    ? chan_in[v] : chan_in[v_anti];
                    }
                }
            }
        } else {
            for (int c = c_begin; c < c_end; ++c) {
                const float* chan_in = in + c * input_N;
                float* chan_out = out + c * output_N;
                for (int v = 0; v < output_N; ++v) {
                    chan_out[v] = (chan_in[v] + chan_in[v ^ anti_mask]) * 0.5f;
                }
            }
        }
    };

    if (use_threads) {
        thread_pool->ForEach(static_cast<size_t>(num_channels),
            [&](size_t, size_t begin, size_t end) {
                do_channels(static_cast<int>(begin), static_cast<int>(end));
            });
    } else {
        do_channels(0, num_channels);
    }
}

void HCNNPool::backward(const float* grad_out, float* grad_in, int num_channels,
                        const std::vector<int>* max_indices) const {
    const uint32_t anti_mask = (1u << input_dim) - 1;
    const bool use_threads = thread_pool && input_dim >= POOL_THREAD_DIM_THRESHOLD;

    auto do_channels = [&](int c_begin, int c_end) {
        // Zero the output range for these channels
        for (int i = c_begin * input_N; i < c_end * input_N; ++i)
            grad_in[i] = 0.0f;

        if (type == PoolType::MAX) {
            if (!max_indices)
                throw std::runtime_error("MAX pool backward requires max_indices from forward pass");
            for (int c = c_begin; c < c_end; ++c) {
                const float* g_out = grad_out + c * output_N;
                float* g_in = grad_in + c * input_N;
                const int* idx = max_indices->data() + c * output_N;
                for (int v = 0; v < output_N; ++v) {
                    g_in[idx[v]] = g_out[v];
                }
            }
        } else {
            for (int c = c_begin; c < c_end; ++c) {
                const float* g_out = grad_out + c * output_N;
                float* g_in = grad_in + c * input_N;
                for (int v = 0; v < output_N; ++v) {
                    g_in[v] += g_out[v] * 0.5f;
                    g_in[v ^ anti_mask] += g_out[v] * 0.5f;
                }
            }
        }
    };

    if (use_threads) {
        thread_pool->ForEach(static_cast<size_t>(num_channels),
            [&](size_t, size_t begin, size_t end) {
                do_channels(static_cast<int>(begin), static_cast<int>(end));
            });
    } else {
        do_channels(0, num_channels);
    }
}

} // namespace hcnn
