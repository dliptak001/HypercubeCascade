// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 David Liptak

#pragma once

#include "HCNNTypes.h"

#include <vector>
#include <cstdint>

namespace hcnn {

class ThreadPool;

/**
 * @class HCNNPool
 * @brief Antipodal pooling layer — private implementation.
 *
 * Not installed.  Use `HCNN::AddPool` from application code.  In-tree tests
 * may include this header.
 *
 * Pairs each vertex `v` with its complement `v ^ (2^DIM - 1)` and reduces
 * DIM by 1 (MAX or AVG).  Stateless — no learnable parameters.
 */
class HCNNPool {
public:
    /// @param input_dim  Hypercube dim before pool; requires 2 <= dim <= 30
    ///                   (N = 2^dim fits in signed 32-bit int; leaves dim >= 1).
    HCNNPool(int input_dim, PoolType type = PoolType::MAX);

    void forward(const float* in, float* out, int num_channels,
                 std::vector<int>* max_indices = nullptr) const;

    void backward(const float* grad_out, float* grad_in, int num_channels,
                  const std::vector<int>* max_indices) const;

    int get_input_dim() const { return input_dim; }
    int get_output_dim() const { return output_dim; }
    int get_input_N() const { return input_N; }
    int get_output_N() const { return output_N; }

    /// Non-owning; pool must outlive any ForEach that uses it.
    void set_thread_pool(ThreadPool* tp) { thread_pool = tp; }

private:
    int input_dim, output_dim, input_N, output_N;
    PoolType type;
    ThreadPool* thread_pool = nullptr;
};

} // namespace hcnn
