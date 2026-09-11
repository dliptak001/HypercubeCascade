// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 David Liptak

#pragma once

#include "HCNNTypes.h"

#include <vector>
#include <cstdint>
#include <random>

namespace hcnn {

/**
 * Loop nest for `grad_in = W^T * grad_logits` (private A/B only).
 * Same math; different memory traffic.  Default `OutputOuter`.
 * Not on `HCNN` — tests may call `set_grad_in_loop` on this class.
 */
enum class ReadoutGradInLoop {
    FeatureOuter,
    OutputOuter
};

/**
 * @class HCNNReadout
 * @brief FLATTEN linear head — private implementation.
 *
 * Not installed.  Owned by `HCNNNetwork` / `HCNN`.  Features = c_final * N_final;
 * raw linear outputs (no softmax).
 *
 * Two backward paths: `backward()` (in-place step) and
 * `compute_gradients()` + `apply_gradients()` (batch accumulate).
 */
class HCNNReadout {
public:
    /// @param num_outputs  Output dimension (classes or regression dims).
    /// @param num_features Flat feature count (typically c_final * N_final).
    HCNNReadout(int num_outputs, int num_features);

    void randomize_weights(float scale, std::mt19937& rng);

    /// @param in   Length `num_features` (channel-major flatten of final map).
    /// @param out  Length `num_outputs`.
    void forward(const float* in, float* out) const;

    /// Gradients + optimizer step.  `grad_in` may be null (first-layer-like).
    void backward(const float* grad_logits, const float* in,
                  float* grad_in, float learning_rate, float momentum = 0.0f,
                  float weight_decay = 0.0f, int timestep = 0);

    /// Write raw weight/bias/input gradients; no weight update.
    /// `grad_in`, `weight_grad`, and `bias_grad` may each be null if unused.
    void compute_gradients(const float* grad_logits, const float* in,
                           float* grad_in, float* weight_grad, float* bias_grad) const;

    /// Apply averaged gradients via the configured optimizer.
    void apply_gradients(const float* weight_grad, const float* bias_grad,
                         float learning_rate, float momentum, float weight_decay = 0.0f,
                         int timestep = 0);

    /// Configure the optimizer. Allocates second-moment buffers for Adam.
    void set_optimizer(OptimizerType type, float beta1 = 0.9f,
                       float beta2 = 0.999f, float eps = 1e-8f);

    /// Select grad_in loop nest (A/B). Default OutputOuter. Does not affect
    /// forward, dW, or optimizer math — only how W^T * grad_logits is formed.
    void set_grad_in_loop(ReadoutGradInLoop loop) { grad_in_loop_ = loop; }
    ReadoutGradInLoop get_grad_in_loop() const { return grad_in_loop_; }

    OptimizerType get_optimizer_type() const { return optimizer_type_; }

    /// Zero first/second moments without changing weights or optimizer type.
    void clear_optimizer_moments();

    int get_num_outputs() const { return num_outputs; }
    int get_num_features() const { return num_features; }

    float* get_weight_data() { return weights.data(); }
    const float* get_weight_data() const { return weights.data(); }
    int get_weight_size() const { return static_cast<int>(weights.size()); }
    float* get_bias_data() { return bias.data(); }
    const float* get_bias_data() const { return bias.data(); }
    int get_bias_size() const { return static_cast<int>(bias.size()); }

private:
    /// Fill grad_in[0..num_features) = W^T * grad_logits using grad_in_loop_.
    void fill_grad_in(const float* grad_logits, float* grad_in) const;

    int num_outputs;
    int num_features;
    std::vector<float> weights;     // [num_outputs * num_features], row = output
    std::vector<float> bias;
    std::vector<float> weight_m;    // first moment (SGD velocity / Adam m)
    std::vector<float> bias_m;
    std::vector<float> weight_m2;   // second moment (Adam only)
    std::vector<float> bias_m2;
    OptimizerType optimizer_type_ = OptimizerType::ADAM;
    float adam_beta1_ = 0.9f, adam_beta2_ = 0.999f, adam_eps_ = 1e-8f;
    ReadoutGradInLoop grad_in_loop_ = ReadoutGradInLoop::OutputOuter;
};

} // namespace hcnn
