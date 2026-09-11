// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 David Liptak

#include "HCNNNetwork.h"
#include "ThreadPool.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <string>

namespace hcnn {

int HCNNNetwork::vertices_at_dim(int dim) {
    // Signed 32-bit int holds 2^30; dim 31+ overflows or is shift-UB on 32-bit int.
    if (dim < 0 || dim > 30) {
        throw std::runtime_error(
            "HCNNNetwork: dimension out of range for N=2^dim (need 0..30), got "
            + std::to_string(dim));
    }
    return static_cast<int>(std::uint32_t{1} << dim);
}

void HCNNNetwork::invalidate_cached_buffers() {
    step_buf_ready_ = false;
    batch_bufs_ready = false;
    infer_bufs_ready = false;
}

HCNNNetwork::HCNNNetwork(int dim, int num_outputs, int input_channels,
                         TaskType task_type, size_t num_threads)
    : start_dim(dim), current_dim(dim), num_outputs(num_outputs),
      input_channels(input_channels),
      task_type_(task_type),
      readout(num_outputs, 1),
      // num_threads: 0 = auto pool, 1 = no background workers (caller-only),
      // N > 1 = N background workers. Single-threaded mode avoids nested
      // oversubscription when the host already parallelizes across networks.
      thread_pool(num_threads == 1
                      ? std::unique_ptr<ThreadPool>{}
                      : std::make_unique<ThreadPool>(num_threads)) {
    // Cap at 30 so N = 2^dim fits in signed 32-bit int (and shift is defined).
    if (dim < 3 || dim > 30) {
        throw std::runtime_error("HCNNNetwork requires 3 <= start_dim <= 30");
    }
    if (input_channels < 1) {
        throw std::runtime_error("HCNNNetwork requires input_channels >= 1");
    }
    if (num_outputs < 1) {
        throw std::runtime_error("HCNNNetwork requires num_outputs >= 1");
    }
    // Loss is fixed by task: Classification → softmax CE, Regression → MSE.
    channel_counts.push_back(input_channels);
}

HCNNNetwork::~HCNNNetwork() = default;

void HCNNNetwork::add_conv(int c_out, Activation activation, bool use_bias,
                           bool use_batchnorm) {
    int c_in = channel_counts.back();
    conv_layers.emplace_back(current_dim, c_in, c_out, activation, use_bias, use_batchnorm);
    conv_layers.back().set_thread_pool(thread_pool.get());
    conv_layers.back().set_optimizer(optimizer_type_, adam_beta1_, adam_beta2_,
                                     adam_eps_);
    channel_counts.push_back(c_out);
    is_conv_layer.push_back(true);
    // Head size / weight blob no longer match the stack until re-randomize.
    weights_initialized_ = false;
    invalidate_cached_buffers();
}

void HCNNNetwork::set_training(bool training) const {
    for (const auto& layer : conv_layers) layer.set_training(training);
}

void HCNNNetwork::set_optimizer(OptimizerType type, float beta1,
                                float beta2, float eps) {
    optimizer_type_ = type;
    adam_beta1_ = beta1;
    adam_beta2_ = beta2;
    adam_eps_ = eps;
    for (auto& layer : conv_layers) layer.set_optimizer(type, beta1, beta2, eps);
    readout.set_optimizer(type, beta1, beta2, eps);
    adam_timestep_ = 0;
}

void HCNNNetwork::reset_optimizer_moments() {
    for (auto& layer : conv_layers) layer.clear_optimizer_moments();
    readout.clear_optimizer_moments();
    adam_timestep_ = 0;
}

void HCNNNetwork::add_pool(PoolType type) {
    // Leave at least dim 1 (N=2). Pooling at dim 0 is shift-UB in HCNNPool.
    if (current_dim < 2) {
        throw std::runtime_error(
            "HCNNNetwork::add_pool: current_dim=" + std::to_string(current_dim)
            + " < 2 (cannot pool further)");
    }
    pool_layers.emplace_back(current_dim, type);
    pool_layers.back().set_thread_pool(thread_pool.get());
    current_dim -= 1;
    channel_counts.push_back(channel_counts.back());
    is_conv_layer.push_back(false);
    // Head size / weight blob no longer match the stack until re-randomize.
    weights_initialized_ = false;
    invalidate_cached_buffers();
}

void HCNNNetwork::randomize_all_weights(float scale, uint64_t seed) {
    int final_channels = channel_counts.back();
    int final_N = vertices_at_dim(current_dim);

    // Preserve A/B grad_in loop; rebuild head to full FLATTEN size.
    const ReadoutGradInLoop gin_loop = readout.get_grad_in_loop();
    readout = HCNNReadout(num_outputs, final_channels * final_N);
    readout.set_grad_in_loop(gin_loop);
    // Re-apply optimizer (fresh HCNNReadout has its own default moments).
    readout.set_optimizer(optimizer_type_, adam_beta1_, adam_beta2_, adam_eps_);

    // Full 64-bit master seed → mt19937. Low-half-only seeds (historical
    // unsigned API) keep the single-arg constructor so bit-identical inits
    // are preserved. Wider seeds expand both halves via seed_seq (high bits
    // are no longer silently dropped).
    std::mt19937 rng = [&]() {
        if ((seed >> 32) == 0ull) {
            return std::mt19937(
                static_cast<std::mt19937::result_type>(seed));
        }
        const std::uint32_t lo = static_cast<std::uint32_t>(seed);
        const std::uint32_t hi = static_cast<std::uint32_t>(seed >> 32);
        std::seed_seq seq{lo, hi};
        return std::mt19937(seq);
    }();
    for (auto& layer : conv_layers) {
        layer.randomize_weights(scale, rng);
    }
    readout.randomize_weights(scale, rng);
    adam_timestep_ = 0;
    // Head size (and any prior PrepareBuffers) may have been wrong before
    // randomize; drop caches so next train/infer reallocates to match.
    invalidate_cached_buffers();
    weights_initialized_ = true;
}

void HCNNNetwork::embed_input(const float* raw_input, int input_length,
                              float* out) const {
    int N = vertices_at_dim(start_dim);
    int total = input_channels * N;
    if (input_length > total) {
        throw std::runtime_error("Input length exceeds capacity ("
                                 + std::to_string(input_channels) + " channels × "
                                 + std::to_string(N) + " vertices = "
                                 + std::to_string(total) + ")");
    }
    for (int i = 0; i < input_length; ++i)
        out[i] = raw_input[i];
    for (int i = input_length; i < total; ++i) {
        out[i] = 0.0f;
    }
}

void HCNNNetwork::forward(const float* first_layer_activations, float* logits) const {
    if (!weights_initialized_) {
        throw std::logic_error(
            "HCNNNetwork::forward: call randomize_all_weights() first "
            "(required after construct or after add_conv/add_pool)");
    }
    if (conv_layers.empty()) {
        throw std::runtime_error("HCNNNetwork::forward called with no conv layers");
    }

    // Inference: force BN to eval mode for the duration of this call and
    // restore the previous per-layer training flag on scope exit.  Caller
    // observes no side effect on BN training state.  RAII restores even on
    // exception.
    EvalModeGuard eval_guard(conv_layers);

    // Compute max buffer size across all layers (including multi-channel input)
    int cur_N = vertices_at_dim(start_dim);
    int max_size = input_channels * cur_N;
    size_t ci = 0, pi = 0;
    for (size_t i = 0; i < is_conv_layer.size(); ++i) {
        if (is_conv_layer[i]) {
            max_size = std::max(max_size, conv_layers[ci].get_c_out() * cur_N);
            ++ci;
        } else {
            cur_N = pool_layers[pi].get_output_N();
            ++pi;
        }
    }

    // Persistent ping-pong scratch — grown on demand, reused across calls.
    if (static_cast<int>(fwd_buf1_.size()) < max_size) {
        fwd_buf1_.resize(max_size);
        fwd_buf2_.resize(max_size);
    }
    float* current  = fwd_buf1_.data();
    float* next_buf = fwd_buf2_.data();

    cur_N = vertices_at_dim(start_dim);
    int input_size = input_channels * cur_N;
    for (int i = 0; i < input_size; ++i) current[i] = first_layer_activations[i];

    ci = 0; pi = 0;

    for (size_t i = 0; i < is_conv_layer.size(); ++i) {
        if (is_conv_layer[i]) {
            conv_layers[ci].forward(current, next_buf);
            ++ci;
        } else {
            pool_layers[pi].forward(current, next_buf, channel_counts[i]);
            cur_N = pool_layers[pi].get_output_N();
            ++pi;
        }
        std::swap(current, next_buf);
    }

    // Final map is channel-major [c * N + v]; FLATTEN treats that buffer as
    // a flat feature vector of length c_final * N_final.
    readout.forward(current, logits);
}

void HCNNNetwork::prepare_inference_buffers() {
    if (infer_bufs_ready) return;

    int N = vertices_at_dim(start_dim);
    int total = input_channels * N;
    size_t nt = thread_pool ? thread_pool->NumThreads() : 1;

    int cur_N = N;
    int max_size = input_channels * cur_N;
    size_t ci0 = 0, pi0 = 0;
    for (size_t i = 0; i < is_conv_layer.size(); ++i) {
        if (is_conv_layer[i]) {
            max_size = std::max(max_size, conv_layers[ci0].get_c_out() * cur_N);
            ++ci0;
        } else {
            cur_N = pool_layers[pi0].get_output_N();
            ++pi0;
        }
    }
    infer_max_layer_size_ = max_size;

    ibufs_.resize(nt);
    for (size_t t = 0; t < nt; ++t) {
        ibufs_[t].buf1.resize(max_size);
        ibufs_[t].buf2.resize(max_size);
        ibufs_[t].embedded.resize(total);
    }
    infer_bufs_ready = true;
}

void HCNNNetwork::forward_batch(const float* flat_inputs, int input_length,
                                int batch_size, float* logits_out) {
    if (!weights_initialized_) {
        throw std::logic_error(
            "HCNNNetwork::forward_batch: call randomize_all_weights() first "
            "(required after construct or after add_conv/add_pool)");
    }
    if (conv_layers.empty()) {
        throw std::runtime_error("HCNNNetwork::forward_batch called with no conv layers");
    }
    if (batch_size <= 0) {
        throw std::invalid_argument("HCNNNetwork::forward_batch: batch_size must be > 0");
    }
    prepare_inference_buffers();

    // Inference: force BN eval mode and restore on exit (see forward()).
    EvalModeGuard eval_guard(conv_layers);

    int total = input_channels * vertices_at_dim(start_dim);
    int K = num_outputs;

    auto process_one = [&](size_t tid, int s) {
        auto& ib = ibufs_[tid];
        std::fill(ib.embedded.begin(), ib.embedded.end(), 0.0f);
        embed_input(flat_inputs + s * input_length, input_length, ib.embedded.data());

        float* current = ib.buf1.data();
        float* next_buf = ib.buf2.data();
        for (int i = 0; i < total; ++i) current[i] = ib.embedded[i];

        size_t ci = 0, pi = 0;
        for (size_t i = 0; i < is_conv_layer.size(); ++i) {
            if (is_conv_layer[i]) {
                conv_layers[ci].forward(current, next_buf);
                ++ci;
            } else {
                pool_layers[pi].forward(current, next_buf, channel_counts[i]);
                ++pi;
            }
            std::swap(current, next_buf);
        }
        readout.forward(current, logits_out + s * K);
    };

    if (thread_pool && batch_size > 1) {
        LayerThreadGuard guard(conv_layers, pool_layers, thread_pool.get());

        thread_pool->ForEach(static_cast<size_t>(batch_size),
            [&](size_t tid, size_t begin, size_t end) {
                for (size_t s = begin; s < end; ++s)
                    process_one(tid, static_cast<int>(s));
            });
    } else {
        for (int s = 0; s < batch_size; ++s)
            process_one(0, s);
    }
}

void HCNNNetwork::train_step(const float* raw_input, int input_length,
                             int target_class, float learning_rate, float momentum,
                             float weight_decay, const float* class_weights) {
    if (task_type_ != TaskType::Classification) {
        throw std::logic_error("HCNNNetwork::train_step: called on a "
                               "Regression task; use train_step_regression");
    }
    if (target_class < 0 || target_class >= num_outputs) {
        throw std::runtime_error("train_step: target_class " + std::to_string(target_class)
                                 + " out of range [0, " + std::to_string(num_outputs) + ")");
    }
    const float cw = (class_weights != nullptr) ? class_weights[target_class] : 1.0f;
    train_step_impl(raw_input, input_length,
        [this, target_class, cw](const float* logits, float* grad_logits_out,
                                 float* probs_scratch) {
            compute_classification_grad(logits, target_class, cw,
                                        probs_scratch, grad_logits_out);
        },
        learning_rate, momentum, weight_decay);
}

// ---------------------------------------------------------------------------
// Mini-batch training: samples are processed in parallel, gradients averaged,
// then a single weight update is applied.
void HCNNNetwork::prepare_all_buffers() {
    prepare_step_buffers();
    prepare_batch_buffers();
    prepare_inference_buffers();
}

// Lazily allocate persistent per-thread buffers on first train_batch call.
// Subsequent calls reuse the same memory — only accumulators are zeroed.
void HCNNNetwork::prepare_batch_buffers() {
    if (batch_bufs_ready) return;

    int N = vertices_at_dim(start_dim);
    int num_layers = static_cast<int>(is_conv_layer.size());
    size_t num_conv = conv_layers.size();
    size_t nt = thread_pool ? thread_pool->NumThreads() : 1;

    // Layer info (sizes are architecture-fixed)
    layer_info_.resize(num_layers + 1);
    layer_info_[0] = {N, input_channels};
    {
        int cur = N;
        size_t ci2 = 0, pi2 = 0;
        for (int i = 0; i < num_layers; ++i) {
            if (is_conv_layer[i]) {
                layer_info_[i + 1] = {cur, conv_layers[ci2].get_c_out()};
                ++ci2;
            } else {
                int out_N = pool_layers[pi2].get_output_N();
                layer_info_[i + 1] = {out_N, layer_info_[i].channels};
                cur = out_N;
                ++pi2;
            }
        }
    }

    // Gradient accumulators
    accum_.resize(nt);
    for (size_t t = 0; t < nt; ++t) {
        auto& a = accum_[t];
        a.conv_kernel_grad.resize(num_conv);
        a.conv_bias_grad.resize(num_conv);
        a.conv_bn_gamma_grad.resize(num_conv);
        a.conv_bn_beta_grad.resize(num_conv);
        a.conv_bn_mean.resize(num_conv);
        a.conv_bn_var.resize(num_conv);
        for (size_t ci = 0; ci < num_conv; ++ci) {
            a.conv_kernel_grad[ci].resize(conv_layers[ci].get_kernel_size());
            a.conv_bias_grad[ci].resize(conv_layers[ci].get_bias_size());
            a.conv_bn_gamma_grad[ci].resize(conv_layers[ci].get_bn_grad_size());
            a.conv_bn_beta_grad[ci].resize(conv_layers[ci].get_bn_grad_size());
            int bn_c = conv_layers[ci].has_batchnorm() ? conv_layers[ci].get_c_out() : 0;
            a.conv_bn_mean[ci].resize(bn_c);
            a.conv_bn_var[ci].resize(bn_c);
        }
        a.readout_weight_grad.resize(readout.get_weight_size());
        a.readout_bias_grad.resize(readout.get_bias_size());
    }

    // Work buffers
    tbufs_.resize(nt);
    for (size_t t = 0; t < nt; ++t) {
        auto& b = tbufs_[t];
        b.cache.resize(num_layers + 1);
        for (int i = 0; i <= num_layers; ++i) {
            auto& li = layer_info_[i];
            b.cache[i].activation.resize(li.channels * li.N);
            b.cache[i].pre_act.resize(li.channels * li.N);
            b.cache[i].max_indices.resize(li.channels * li.N);
        }
        b.logits.resize(num_outputs);
        b.probs.resize(num_outputs);
        b.grad_logits.resize(num_outputs);
        int max_layer_size = 0;
        for (int i = 0; i <= num_layers; ++i)
            max_layer_size = std::max(max_layer_size,
                                      layer_info_[i].channels * layer_info_[i].N);
        b.grad_a.resize(max_layer_size);
        b.grad_b.resize(max_layer_size);
        b.rw_grad.resize(readout.get_weight_size());
        b.rb_grad.resize(readout.get_bias_size());
        b.kg.resize(num_conv);
        b.bg.resize(num_conv);
        b.bn_gg.resize(num_conv);
        b.bn_bg.resize(num_conv);
        b.bn_save.resize(num_conv);
        for (size_t ci = 0; ci < num_conv; ++ci) {
            b.kg[ci].resize(conv_layers[ci].get_kernel_size());
            b.bg[ci].resize(conv_layers[ci].get_bias_size());
            b.bn_gg[ci].resize(conv_layers[ci].get_bn_grad_size());
            b.bn_bg[ci].resize(conv_layers[ci].get_bn_grad_size());
            b.bn_save[ci].resize(conv_layers[ci].get_bn_save_size());
        }
        // Work buffer for HCNNConv::compute_gradients (avoid per-call heap allocs)
        b.conv_work.resize(max_layer_size); // c_out*N, max_layer_size >= that
    }

    batch_bufs_ready = true;
}

void HCNNNetwork::zero_accumulators() {
    for (auto& a : accum_) {
        for (auto& v : a.conv_kernel_grad) std::fill(v.begin(), v.end(), 0.0f);
        for (auto& v : a.conv_bias_grad)   std::fill(v.begin(), v.end(), 0.0f);
        for (auto& v : a.conv_bn_gamma_grad) std::fill(v.begin(), v.end(), 0.0f);
        for (auto& v : a.conv_bn_beta_grad)  std::fill(v.begin(), v.end(), 0.0f);
        for (auto& v : a.conv_bn_mean)       std::fill(v.begin(), v.end(), 0.0f);
        for (auto& v : a.conv_bn_var)        std::fill(v.begin(), v.end(), 0.0f);
        std::fill(a.readout_weight_grad.begin(), a.readout_weight_grad.end(), 0.0f);
        std::fill(a.readout_bias_grad.begin(), a.readout_bias_grad.end(), 0.0f);
    }
}

void HCNNNetwork::train_batch(const float* flat_inputs, int input_length,
                              const int* targets, int batch_size,
                              float learning_rate, float momentum,
                              float weight_decay, const float* class_weights) {
    if (task_type_ != TaskType::Classification) {
        throw std::logic_error("HCNNNetwork::train_batch: called on a "
                               "Regression task; use train_batch_regression");
    }
    if (batch_size <= 0) {
        throw std::invalid_argument("HCNNNetwork::train_batch: batch_size must be > 0");
    }
    for (int i = 0; i < batch_size; ++i) {
        if (targets[i] < 0 || targets[i] >= num_outputs) {
            throw std::runtime_error("train_batch: target[" + std::to_string(i) + "] = "
                                     + std::to_string(targets[i]) + " out of range [0, "
                                     + std::to_string(num_outputs) + ")");
        }
    }
    train_batch_impl(flat_inputs, input_length, batch_size,
        [this, targets, class_weights](int sample_idx, const float* logits,
                                       float* grad_logits_out, float* probs_scratch) {
            float cw = (class_weights != nullptr)
                           ? class_weights[targets[sample_idx]] : 1.0f;
            compute_classification_grad(logits, targets[sample_idx], cw,
                                        probs_scratch, grad_logits_out);
        },
        learning_rate, momentum, weight_decay);
}

// ---------------------------------------------------------------------------
//  Loss-gradient helpers (task-fixed: CE or MSE)
// ---------------------------------------------------------------------------
//
// The readout is loss-agnostic; all loss semantics live here.
// ---------------------------------------------------------------------------

void HCNNNetwork::compute_classification_grad(const float* logits,
                                              int target_class,
                                              float class_weight,
                                              float* probs_scratch,
                                              float* grad_logits_out) const {
    if (task_type_ != TaskType::Classification) {
        throw std::logic_error("compute_classification_grad: called on a "
                               "Regression task");
    }
    // Softmax + cross-entropy:
    //   p[i] = exp(logits[i] - max) / sum(exp(...))
    //   dL/d(logits[i]) = p[i] - 1[i == target]
    float max_logit = logits[0];
    for (int i = 1; i < num_outputs; ++i) {
        if (logits[i] > max_logit) max_logit = logits[i];
    }
    float sum_exp = 0.0f;
    for (int i = 0; i < num_outputs; ++i) {
        probs_scratch[i] = std::exp(logits[i] - max_logit);
        sum_exp += probs_scratch[i];
    }
    // Pathological all-underflow: fall back to uniform (avoid 0/0).
    if (!(sum_exp > 0.0f)) {
        const float inv_k = 1.0f / static_cast<float>(num_outputs);
        for (int i = 0; i < num_outputs; ++i) probs_scratch[i] = inv_k;
    } else {
        for (int i = 0; i < num_outputs; ++i) probs_scratch[i] /= sum_exp;
    }
    for (int i = 0; i < num_outputs; ++i) {
        grad_logits_out[i] = class_weight *
            (probs_scratch[i] - (i == target_class ? 1.0f : 0.0f));
    }
}

void HCNNNetwork::compute_regression_grad(const float* logits,
                                          const float* target,
                                          float* grad_logits_out) const {
    if (task_type_ != TaskType::Regression) {
        throw std::logic_error("compute_regression_grad: called on a "
                               "Classification task");
    }
    // Sum-style MSE for one sample: dL/d(pred[i]) = pred[i] - target[i]
    // (LR absorbs the usual 2/K mean-MSE scale).
    for (int i = 0; i < num_outputs; ++i) {
        grad_logits_out[i] = logits[i] - target[i];
    }
}

// ---------------------------------------------------------------------------
//  Regression training — thin wrappers over the shared training cores
// ---------------------------------------------------------------------------
void HCNNNetwork::train_step_regression(const float* raw_input, int input_length,
                                        const float* target, float learning_rate,
                                        float momentum, float weight_decay) {
    if (task_type_ != TaskType::Regression) {
        throw std::logic_error("HCNNNetwork::train_step_regression: called "
                               "on a Classification task; use train_step");
    }
    train_step_impl(raw_input, input_length,
        [this, target](const float* logits, float* grad_logits_out,
                       float* /*probs_scratch*/) {
            compute_regression_grad(logits, target, grad_logits_out);
        },
        learning_rate, momentum, weight_decay);
}

void HCNNNetwork::train_batch_regression(const float* flat_inputs,
                                         int input_length,
                                         const float* flat_targets,
                                         int batch_size,
                                         float learning_rate, float momentum,
                                         float weight_decay) {
    if (task_type_ != TaskType::Regression) {
        throw std::logic_error("HCNNNetwork::train_batch_regression: called "
                               "on a Classification task; use train_batch");
    }
    if (batch_size <= 0) {
        throw std::invalid_argument("HCNNNetwork::train_batch_regression: "
                                    "batch_size must be > 0");
    }
    const int K = num_outputs;
    train_batch_impl(flat_inputs, input_length, batch_size,
        [this, flat_targets, K](int sample_idx, const float* logits,
                        float* grad_logits_out, float* /*probs_scratch*/) {
            compute_regression_grad(logits, flat_targets + sample_idx * K,
                                    grad_logits_out);
        },
        learning_rate, momentum, weight_decay);
}

// ---------------------------------------------------------------------------
//  Shared training cores
// ---------------------------------------------------------------------------
//
// train_step_impl and train_batch_impl are the single source of truth for
// the forward / backward / weight-update pipeline.  The four public
// train_* methods are thin wrappers that validate their task-specific
// arguments, build a small loss-grad lambda, and delegate here.  Adding a
// new task type or loss function does not require touching these bodies.
// ---------------------------------------------------------------------------

void HCNNNetwork::prepare_step_buffers() {
    if (step_buf_ready_) return;

    const int N = vertices_at_dim(start_dim);
    const int total = input_channels * N;
    const int num_layers = static_cast<int>(is_conv_layer.size());

    step_buf_.embedded.resize(total);
    step_buf_.cache.resize(num_layers + 1);
    step_buf_.layer_N.resize(num_layers + 1);
    step_buf_.layer_ch.resize(num_layers + 1);

    step_buf_.layer_N[0] = N;
    step_buf_.layer_ch[0] = input_channels;
    step_buf_.cache[0].activation.resize(total);

    int cur_N = N;
    int cur_ch = input_channels;
    size_t ci = 0, pi = 0;
    int max_act_size = total;

    for (int i = 0; i < num_layers; ++i) {
        auto& c = step_buf_.cache[i + 1];
        if (is_conv_layer[i]) {
            cur_ch = conv_layers[ci].get_c_out();
            int sz = cur_ch * cur_N;
            c.activation.resize(sz);
            c.pre_act.resize(sz);
            c.bn_save.resize(conv_layers[ci].get_bn_save_size());
            if (sz > max_act_size) max_act_size = sz;
            ++ci;
        } else {
            int out_N = pool_layers[pi].get_output_N();
            int sz = cur_ch * out_N;
            c.activation.resize(sz);
            c.max_indices.resize(sz);
            cur_N = out_N;
            if (sz > max_act_size) max_act_size = sz;
            ++pi;
        }
        step_buf_.layer_N[i + 1] = cur_N;
        step_buf_.layer_ch[i + 1] = cur_ch;
    }

    step_buf_.logits.resize(num_outputs);
    step_buf_.probs.resize(num_outputs);
    step_buf_.grad_logits.resize(num_outputs);
    step_buf_.grad_a.resize(max_act_size);
    step_buf_.grad_b.resize(max_act_size);

    step_buf_ready_ = true;
}

void HCNNNetwork::train_step_impl(const float* raw_input, int input_length,
                                  const LossGradStepFn& loss_grad,
                                  float learning_rate, float momentum,
                                  float weight_decay) {
    if (!weights_initialized_) {
        throw std::logic_error(
            "HCNNNetwork::train_step: call randomize_all_weights() first "
            "(required after construct or after add_conv/add_pool)");
    }
    if (conv_layers.empty()) {
        throw std::runtime_error("HCNNNetwork::train_step_impl called with no conv layers");
    }
    prepare_step_buffers();

    const int num_layers = static_cast<int>(is_conv_layer.size());
    ++adam_timestep_;

    set_training(true);

    std::fill(step_buf_.embedded.begin(), step_buf_.embedded.end(), 0.0f);
    embed_input(raw_input, input_length, step_buf_.embedded.data());

    auto& cache = step_buf_.cache;
    std::copy(step_buf_.embedded.begin(), step_buf_.embedded.end(),
              cache[0].activation.begin());

    size_t ci = 0, pi = 0;

    for (int i = 0; i < num_layers; ++i) {
        auto& c = cache[i + 1];
        if (is_conv_layer[i]) {
            conv_layers[ci].forward(cache[i].activation.data(),
                                    c.activation.data(), c.pre_act.data(),
                                    c.bn_save.empty() ? nullptr : c.bn_save.data());
            ++ci;
        } else {
            pool_layers[pi].forward(cache[i].activation.data(),
                                    c.activation.data(),
                                    step_buf_.layer_ch[i],
                                    &c.max_indices);
            ++pi;
        }
    }

    std::fill(step_buf_.logits.begin(), step_buf_.logits.end(), 0.0f);
    readout.forward(cache[num_layers].activation.data(),
                    step_buf_.logits.data());

    loss_grad(step_buf_.logits.data(), step_buf_.grad_logits.data(),
              step_buf_.probs.data());

    int final_sz = step_buf_.layer_ch[num_layers] * step_buf_.layer_N[num_layers];
    std::fill(step_buf_.grad_a.begin(), step_buf_.grad_a.begin() + final_sz, 0.0f);
    readout.backward(step_buf_.grad_logits.data(),
                     cache[num_layers].activation.data(),
                     step_buf_.grad_a.data(), learning_rate, momentum,
                     weight_decay, adam_timestep_);

    ci = conv_layers.size();
    pi = pool_layers.size();

    float* grad_cur = step_buf_.grad_a.data();
    float* grad_prev = step_buf_.grad_b.data();

    for (int i = num_layers - 1; i >= 0; --i) {
        int prev_sz = step_buf_.layer_ch[i] * step_buf_.layer_N[i];
        std::fill(grad_prev, grad_prev + prev_sz, 0.0f);

        if (is_conv_layer[i]) {
            --ci;
            conv_layers[ci].backward(grad_cur,
                                     cache[i].activation.data(),
                                     cache[i + 1].pre_act.data(),
                                     (i > 0) ? grad_prev : nullptr,
                                     learning_rate, momentum, weight_decay,
                                     cache[i + 1].bn_save.empty() ? nullptr
                                         : cache[i + 1].bn_save.data(),
                                     adam_timestep_,
                                     cache[i + 1].activation.data());
        } else {
            --pi;
            pool_layers[pi].backward(grad_cur, grad_prev,
                                     step_buf_.layer_ch[i],
                                     &cache[i + 1].max_indices);
        }

        std::swap(grad_cur, grad_prev);
    }
}

void HCNNNetwork::train_batch_impl(const float* flat_inputs, int input_length,
                                   int batch_size,
                                   const LossGradBatchFn& loss_grad,
                                   float learning_rate, float momentum,
                                   float weight_decay) {
    if (!weights_initialized_) {
        throw std::logic_error(
            "HCNNNetwork::train_batch: call randomize_all_weights() first "
            "(required after construct or after add_conv/add_pool)");
    }
    if (conv_layers.empty()) {
        throw std::runtime_error("HCNNNetwork::train_batch_impl called with no conv layers");
    }
    prepare_batch_buffers();
    zero_accumulators();
    set_training(true);
    ++adam_timestep_;

    int num_layers = static_cast<int>(is_conv_layer.size());
    size_t num_conv = conv_layers.size();
    size_t nt = accum_.size();

    // Suppress per-sample running-stats updates for BN layers (avoid data race).
    // RAII guard restores on scope exit, including on exception.
    BNStatsGuard bn_guard(conv_layers);

    // Process samples in parallel, each thread accumulates into its own accum
    auto process_sample = [&](size_t tid, int sample_idx) {
        auto& a = accum_[tid];
        auto& b = tbufs_[tid];

        // Embed directly into cache[0]
        {
            auto& c0 = b.cache[0];
            std::fill(c0.activation.begin(), c0.activation.end(), 0.0f);
            embed_input(flat_inputs + sample_idx * input_length, input_length,
                        c0.activation.data());
        }

        size_t ci = 0, pi = 0;

        // Forward pass
        for (int i = 0; i < num_layers; ++i) {
            auto& c = b.cache[i + 1];
            if (is_conv_layer[i]) {
                conv_layers[ci].forward(b.cache[i].activation.data(),
                                        c.activation.data(), c.pre_act.data(),
                                        b.bn_save[ci].empty() ? nullptr
                                            : b.bn_save[ci].data());
                // Accumulate per-sample BN mean/var for deferred running-stats update
                if (!a.conv_bn_mean[ci].empty()) {
                    int c_out = conv_layers[ci].get_c_out();
                    for (int co = 0; co < c_out; ++co) {
                        a.conv_bn_mean[ci][co] += b.bn_save[ci][c_out + co];
                        a.conv_bn_var[ci][co]  += b.bn_save[ci][2 * c_out + co];
                    }
                }
                ++ci;
            } else {
                pool_layers[pi].forward(b.cache[i].activation.data(),
                                        c.activation.data(),
                                        layer_info_[i].channels,
                                        &c.max_indices);
                ++pi;
            }
        }

        // Readout forward
        auto& final_c = b.cache[num_layers];
        std::fill(b.logits.begin(), b.logits.end(), 0.0f);
        readout.forward(final_c.activation.data(), b.logits.data());

        // Loss gradient (provided by caller).  b.probs is per-thread scratch
        // for classification softmax; regression callbacks ignore it.
        loss_grad(sample_idx, b.logits.data(), b.grad_logits.data(), b.probs.data());

        // Readout backward
        int final_size = layer_info_[num_layers].channels * layer_info_[num_layers].N;
        std::fill(b.grad_a.begin(), b.grad_a.begin() + final_size, 0.0f);
        readout.compute_gradients(b.grad_logits.data(), final_c.activation.data(),
                                  b.grad_a.data(),
                                  b.rw_grad.data(), b.rb_grad.data());

        for (int i = 0; i < readout.get_weight_size(); ++i)
            a.readout_weight_grad[i] += b.rw_grad[i];
        for (int i = 0; i < readout.get_bias_size(); ++i)
            a.readout_bias_grad[i] += b.rb_grad[i];

        // Backward through layers — ping-pong between grad_a and grad_b
        ci = conv_layers.size();
        pi = pool_layers.size();

        for (int i = num_layers - 1; i >= 0; --i) {
            int prev_size = layer_info_[i].channels * layer_info_[i].N;
            std::fill(b.grad_b.begin(), b.grad_b.begin() + prev_size, 0.0f);

            if (is_conv_layer[i]) {
                --ci;
                conv_layers[ci].compute_gradients(
                    b.grad_a.data(),
                    b.cache[i].activation.data(),
                    b.cache[i + 1].pre_act.data(),
                    (i > 0) ? b.grad_b.data() : nullptr,
                    b.kg[ci].data(),
                    b.bg[ci].empty() ? nullptr : b.bg[ci].data(),
                    b.conv_work.data(),
                    b.bn_save[ci].empty() ? nullptr : b.bn_save[ci].data(),
                    b.bn_gg[ci].empty() ? nullptr : b.bn_gg[ci].data(),
                    b.bn_bg[ci].empty() ? nullptr : b.bn_bg[ci].data(),
                    b.cache[i + 1].activation.data());

                for (int j = 0; j < conv_layers[ci].get_kernel_size(); ++j)
                    a.conv_kernel_grad[ci][j] += b.kg[ci][j];
                for (int j = 0; j < conv_layers[ci].get_bias_size(); ++j)
                    a.conv_bias_grad[ci][j] += b.bg[ci][j];
                for (int j = 0; j < conv_layers[ci].get_bn_grad_size(); ++j)
                    a.conv_bn_gamma_grad[ci][j] += b.bn_gg[ci][j];
                for (int j = 0; j < conv_layers[ci].get_bn_grad_size(); ++j)
                    a.conv_bn_beta_grad[ci][j] += b.bn_bg[ci][j];
            } else {
                --pi;
                pool_layers[pi].backward(b.grad_a.data(), b.grad_b.data(),
                                         layer_info_[i].channels,
                                         &b.cache[i + 1].max_indices);
            }

            std::swap(b.grad_a, b.grad_b);
        }
    };

    if (thread_pool && batch_size > 1) {
        LayerThreadGuard guard(conv_layers, pool_layers, thread_pool.get());

        thread_pool->ForEach(static_cast<size_t>(batch_size),
            [&](size_t tid, size_t begin, size_t end) {
                for (size_t s = begin; s < end; ++s)
                    process_sample(tid, static_cast<int>(s));
            });
    } else {
        for (int s = 0; s < batch_size; ++s)
            process_sample(0, s);
    }

    // Reduce across threads and average
    float scale = 1.0f / static_cast<float>(batch_size);

    for (size_t ci = 0; ci < num_conv; ++ci) {
        int ks = conv_layers[ci].get_kernel_size();
        int bs = conv_layers[ci].get_bias_size();
        int bns = conv_layers[ci].get_bn_grad_size();
        auto& base_kg = accum_[0].conv_kernel_grad[ci];
        auto& base_bg = accum_[0].conv_bias_grad[ci];
        auto& base_bng = accum_[0].conv_bn_gamma_grad[ci];
        auto& base_bnb = accum_[0].conv_bn_beta_grad[ci];

        for (size_t t = 1; t < nt; ++t) {
            for (int j = 0; j < ks; ++j) base_kg[j] += accum_[t].conv_kernel_grad[ci][j];
            for (int j = 0; j < bs; ++j) base_bg[j] += accum_[t].conv_bias_grad[ci][j];
            for (int j = 0; j < bns; ++j) base_bng[j] += accum_[t].conv_bn_gamma_grad[ci][j];
            for (int j = 0; j < bns; ++j) base_bnb[j] += accum_[t].conv_bn_beta_grad[ci][j];
        }

        for (int j = 0; j < ks; ++j) base_kg[j] *= scale;
        for (int j = 0; j < bs; ++j) base_bg[j] *= scale;
        for (int j = 0; j < bns; ++j) base_bng[j] *= scale;
        for (int j = 0; j < bns; ++j) base_bnb[j] *= scale;

        conv_layers[ci].apply_gradients(base_kg.data(),
                                        base_bg.empty() ? nullptr : base_bg.data(),
                                        learning_rate, momentum, weight_decay,
                                        base_bng.empty() ? nullptr : base_bng.data(),
                                        base_bnb.empty() ? nullptr : base_bnb.data(),
                                        adam_timestep_);

        // Reduce per-sample BN stats and update running mean/var (race-free)
        if (conv_layers[ci].has_batchnorm()) {
            int c_out = conv_layers[ci].get_c_out();
            auto& base_mean = accum_[0].conv_bn_mean[ci];
            auto& base_var  = accum_[0].conv_bn_var[ci];
            for (size_t t = 1; t < nt; ++t) {
                for (int j = 0; j < c_out; ++j) {
                    base_mean[j] += accum_[t].conv_bn_mean[ci][j];
                    base_var[j]  += accum_[t].conv_bn_var[ci][j];
                }
            }
            for (int j = 0; j < c_out; ++j) {
                base_mean[j] *= scale;
                base_var[j]  *= scale;
            }
            conv_layers[ci].update_running_stats(base_mean.data(), base_var.data());
        }
    }

    auto& base_rw = accum_[0].readout_weight_grad;
    auto& base_rb = accum_[0].readout_bias_grad;
    for (size_t t = 1; t < nt; ++t) {
        for (size_t j = 0; j < base_rw.size(); ++j) base_rw[j] += accum_[t].readout_weight_grad[j];
        for (size_t j = 0; j < base_rb.size(); ++j) base_rb[j] += accum_[t].readout_bias_grad[j];
    }
    for (auto& g : base_rw) g *= scale;
    for (auto& g : base_rb) g *= scale;

    readout.apply_gradients(base_rw.data(), base_rb.data(), learning_rate, momentum,
                            weight_decay, adam_timestep_);
}

} // namespace hcnn
