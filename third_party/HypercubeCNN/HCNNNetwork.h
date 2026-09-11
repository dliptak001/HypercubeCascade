// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 David Liptak

#pragma once

// =============================================================================
// PRIVATE IMPLEMENTATION — not part of the installed SDK.
//
// Boundary policy:
//   1. This header (and Conv/Pool/Readout/ThreadPool) is never installed.
//      Only HCNN.cpp and in-tree tests include it.
//   2. Not a second public API.  Apps use hcnn::HCNN only.
//   3. User-facing features land on HCNN first; this class gains only what
//      the facade needs (no parallel public knobs).
// =============================================================================

#include "HCNNTypes.h"
#include "HCNNConv.h"
#include "HCNNPool.h"
#include "HCNNReadout.h"
#include <functional>
#include <memory>
#include <vector>
#include <stdexcept>

namespace hcnn {

class ThreadPool;
class HCNN;  // sole public owner (PIMPL)

/**
 * @class HCNNNetwork
 * @brief Private pipeline orchestrator owned by `HCNN` (PIMPL).
 *
 * Not installed.  Not for application code.  In-tree tests may include this
 * header to exercise lifecycle / layer contracts.
 *
 * Owns conv/pool stacks, FLATTEN readout, ThreadPool, and train/infer scratch.
 * Non-copyable, non-movable (live worker threads).  Instance is exclusive-use:
 * host must not call concurrently on the same network.
 */
class HCNNNetwork {
    friend class HCNN;

public:
    /// @param num_threads 0 = auto pool, 1 = single-threaded (no workers),
    ///        N > 1 = N background workers. See HCNN constructor.
    /// @param start_dim Hypercube dimension in [3, 30] (N = 2^dim fits in int).
    HCNNNetwork(int start_dim, int num_outputs = 10,
                int input_channels = 1,
                TaskType task_type = TaskType::Classification,
                size_t num_threads = 0);
    ~HCNNNetwork();

    HCNNNetwork(const HCNNNetwork&) = delete;
    HCNNNetwork& operator=(const HCNNNetwork&) = delete;
    HCNNNetwork(HCNNNetwork&&) = delete;
    HCNNNetwork& operator=(HCNNNetwork&&) = delete;

    /// Clears weights_initialized_ (call randomize_all_weights before train/infer).
    void add_conv(int c_out, Activation activation = Activation::RELU,
                  bool use_bias = true, bool use_batchnorm = false);
    /// Antipodal pool; requires current_dim >= 2.  Clears weights_initialized_.
    void add_pool(PoolType type = PoolType::MAX);

    /// Set training mode (true) or eval mode (false) for all layers with BN.
    void set_training(bool training) const;

    /// Configure the optimizer for all layers (and any layers added later).
    /// Resets the global Adam timestep. Survives randomize_all_weights.
    void set_optimizer(OptimizerType type, float beta1 = 0.9f,
                       float beta2 = 0.999f, float eps = 1e-8f);

    /// Initialize all weights.  scale > 0: uniform [-scale, +scale].
    /// scale <= 0 (default): Xavier/Glorot uniform per layer.
    /// Rebuilds the FLATTEN readout to c_final * N_final; preserves
    /// optimizer + grad_in loop; invalidates cached work buffers.
    /// Initialize weights from a full 64-bit master seed.
    /// Seeds with high half zero keep the historical mt19937(seed32) path.
    void randomize_all_weights(float scale = 0.0f, uint64_t seed = 42);

    void embed_input(const float* raw_input, int input_length,
                     float* first_layer_activations) const;

    /// Single-sample forward pass.  Reuses persistent ping-pong scratch —
    /// no per-call allocation in steady state.  Not safe concurrent with
    /// other forward() / train_* on the same instance.
    void forward(const float* first_layer_activations, float* logits) const;

    /// Batch inference: embed + forward for multiple samples in parallel.
    /// `flat_inputs` is `batch_size * input_length` floats (contiguous,
    /// row-major).  `logits_out` must have `batch_size * num_outputs` floats.
    /// For classification nets these are raw (pre-softmax) logits; for
    /// regression nets they are the raw real-valued predictions.
    void forward_batch(const float* flat_inputs, int input_length,
                       int batch_size, float* logits_out);

    void train_step(const float* raw_input, int input_length,
                    int target_class, float learning_rate, float momentum = 0.0f,
                    float weight_decay = 0.0f,
                    const float* class_weights = nullptr);

    /// Mini-batch training: process batch_size samples in parallel (when a
    /// ThreadPool is present and batch_size > 1), average gradients, then
    /// apply a single weight update.  Serial path if no pool or batch_size==1.
    /// `flat_inputs` is `batch_size * input_length` contiguous floats.
    /// class_weights: optional per-class loss scaling (length num_outputs).
    /// Classification only — throws std::logic_error if task_type_ is
    /// Regression.
    void train_batch(const float* flat_inputs, int input_length,
                     const int* targets, int batch_size,
                     float learning_rate, float momentum = 0.0f,
                     float weight_decay = 0.0f,
                     const float* class_weights = nullptr);

    /// Regression counterpart of train_step.  `target` is a pointer to
    /// `num_outputs` floats — the per-output regression targets for a
    /// single sample.  Throws std::logic_error if task_type_ is
    /// Classification.
    void train_step_regression(const float* raw_input, int input_length,
                               const float* target, float learning_rate,
                               float momentum = 0.0f,
                               float weight_decay = 0.0f);

    /// Regression counterpart of train_batch.  `flat_targets` is
    /// `batch_size * num_outputs` contiguous floats (row-major).
    /// Throws std::logic_error if task_type_ is Classification.
    void train_batch_regression(const float* flat_inputs, int input_length,
                                const float* flat_targets, int batch_size,
                                float learning_rate, float momentum = 0.0f,
                                float weight_decay = 0.0f);

    int get_start_dim() const { return start_dim; }
    int get_start_N() const { return vertices_at_dim(start_dim); }
    int get_current_dim() const { return current_dim; }
    int get_input_channels() const { return input_channels; }
    int get_num_outputs() const { return num_outputs; }
    TaskType get_task_type() const { return task_type_; }
    OptimizerType get_optimizer_type() const { return optimizer_type_; }

    /// Zero all layer optimizer moments and reset the Adam timestep to 0.
    void reset_optimizer_moments();

    /// True after a successful randomize_all_weights (full FLATTEN head sized).
    bool weights_initialized() const { return weights_initialized_; }

    HCNNConv& get_conv(size_t i) { return conv_layers[i]; }
    const HCNNConv& get_conv(size_t i) const { return conv_layers[i]; }
    HCNNReadout& get_readout() { return readout; }
    const HCNNReadout& get_readout() const { return readout; }
    size_t get_num_conv() const { return conv_layers.size(); }
    size_t get_num_pool() const { return pool_layers.size(); }
    const std::vector<bool>& get_layer_types() const { return is_conv_layer; }
    const std::vector<int>& get_channel_counts() const { return channel_counts; }

    /// Eagerly allocate all internal work buffers (single-step, batch,
    /// inference).  Each is idempotent — safe to call multiple times.
    /// Prefer calling after randomize_all_weights so readout-sized buffers match.
    void prepare_all_buffers();

private:
    /// N = 2^dim for dim in [0, 30] (fits in signed 32-bit int).
    static int vertices_at_dim(int dim);

    /// Drop lazy step/batch/inference buffer caches (arch or head changed).
    void invalidate_cached_buffers();

    int start_dim;
    int current_dim;
    int num_outputs;
    int input_channels;
    TaskType task_type_;
    int adam_timestep_{0};     // Global optimizer timestep (incremented per train_step/train_batch)
    OptimizerType optimizer_type_ = OptimizerType::ADAM;
    float adam_beta1_ = 0.9f, adam_beta2_ = 0.999f, adam_eps_ = 1e-8f;
    bool weights_initialized_{false};  // true after randomize_all_weights
    std::vector<HCNNConv> conv_layers;
    std::vector<HCNNPool> pool_layers;
    std::vector<bool> is_conv_layer;
    std::vector<int> channel_counts;
    HCNNReadout readout;
    std::unique_ptr<ThreadPool> thread_pool;

    // --- Persistent batch-training buffers (allocated once, reused every train_batch) ---
    struct LayerInfo { int N; int channels; };

    struct ThreadAccum {
        std::vector<std::vector<float>> conv_kernel_grad;
        std::vector<std::vector<float>> conv_bias_grad;
        std::vector<std::vector<float>> conv_bn_gamma_grad;
        std::vector<std::vector<float>> conv_bn_beta_grad;
        std::vector<std::vector<float>> conv_bn_mean;   // per-conv BN mean accumulator
        std::vector<std::vector<float>> conv_bn_var;    // per-conv BN var accumulator
        std::vector<float> readout_weight_grad;
        std::vector<float> readout_bias_grad;
    };

    struct ThreadBuf {
        struct LayerCache {
            std::vector<float> activation;
            std::vector<float> pre_act;
            std::vector<int> max_indices;
        };
        std::vector<LayerCache> cache;
        std::vector<float> logits, probs, grad_logits;
        std::vector<float> grad_a, grad_b;
        std::vector<float> rw_grad, rb_grad;
        std::vector<std::vector<float>> kg, bg;
        std::vector<std::vector<float>> bn_gg, bn_bg;  // per-conv BN gamma/beta grads
        std::vector<std::vector<float>> bn_save;        // per-conv BN inv_std cache
        std::vector<float> conv_work;     // work buf for HCNNConv::compute_gradients
    };

    bool batch_bufs_ready{false};
    std::vector<LayerInfo> layer_info_;
    std::vector<ThreadAccum> accum_;
    std::vector<ThreadBuf> tbufs_;

    void prepare_batch_buffers();
    void zero_accumulators();

    // ----- Shared training cores -----------------------------------------
    //
    // train_step / train_step_regression and train_batch / train_batch_regression
    // share an identical forward / backward / weight-update pipeline.  The
    // only thing that differs between classification and regression is the
    // computation of dL/d(logits) for each sample.  We capture that one
    // step in a callable and let the wrappers (the public train_* methods)
    // build the appropriate lambda.
    //
    // The "loss-grad" callbacks receive a `probs_scratch` buffer of size
    // num_outputs floats.  Classification lambdas use it to hold the
    // softmax probabilities (a side effect of compute_classification_grad);
    // regression lambdas ignore it.
    using LossGradStepFn = std::function<void(const float* logits,
                                              float* grad_logits_out,
                                              float* probs_scratch)>;
    using LossGradBatchFn = std::function<void(int sample_idx,
                                               const float* logits,
                                               float* grad_logits_out,
                                               float* probs_scratch)>;

    // Single-sample training core.  Caller is responsible for any task /
    // target validation -- this method trusts its inputs.
    void train_step_impl(const float* raw_input, int input_length,
                         const LossGradStepFn& loss_grad,
                         float learning_rate, float momentum,
                         float weight_decay);

    // Mini-batch training core.  Caller is responsible for any task /
    // target / batch_size validation.
    void train_batch_impl(const float* flat_inputs, int input_length,
                          int batch_size,
                          const LossGradBatchFn& loss_grad,
                          float learning_rate, float momentum,
                          float weight_decay);

    // Softmax CE: dL/d(logits).  `probs_scratch` receives softmax as a side
    // effect.  Throws if task is not Classification.
    void compute_classification_grad(const float* logits, int target_class,
                                     float class_weight,
                                     float* probs_scratch,
                                     float* grad_logits_out) const;

    // Sum-style MSE: dL/d(pred[i]) = pred[i] - target[i].  Throws if task
    // is not Regression.
    void compute_regression_grad(const float* logits, const float* target,
                                 float* grad_logits_out) const;

    // --- Persistent inference buffers (allocated once, reused every forward_batch) ---
    struct InferenceBuf {
        std::vector<float> buf1, buf2;
        std::vector<float> embedded;
    };
    bool infer_bufs_ready{false};
    std::vector<InferenceBuf> ibufs_;
    int infer_max_layer_size_{0};

    void prepare_inference_buffers();

    // Persistent scratch for single-sample forward() — sized to the largest
    // layer in the network and reused across calls.
    mutable std::vector<float> fwd_buf1_;
    mutable std::vector<float> fwd_buf2_;

    // --- Persistent single-step training buffers (allocated once, reused every train_step) ---
    struct StepCache {
        std::vector<float> activation;
        std::vector<float> pre_act;
        std::vector<float> bn_save;
        std::vector<int> max_indices;
    };
    struct StepBuf {
        std::vector<float> embedded;
        std::vector<StepCache> cache;
        std::vector<int> layer_N;
        std::vector<int> layer_ch;
        std::vector<float> logits, probs, grad_logits;
        std::vector<float> grad_a, grad_b;
    };
    bool step_buf_ready_{false};
    StepBuf step_buf_;
    void prepare_step_buffers();

    // RAII guard to disable per-layer threading during batch dispatch
    // and restore it when the scope exits (including on exception).
    struct LayerThreadGuard {
        std::vector<HCNNConv>& conv;
        std::vector<HCNNPool>& pool_layers;
        ThreadPool* pool;
        LayerThreadGuard(std::vector<HCNNConv>& c, std::vector<HCNNPool>& p, ThreadPool* tp)
            : conv(c), pool_layers(p), pool(tp) {
            for (auto& layer : conv) layer.set_thread_pool(nullptr);
            for (auto& layer : pool_layers) layer.set_thread_pool(nullptr);
        }
        ~LayerThreadGuard() {
            for (auto& layer : conv) layer.set_thread_pool(pool);
            for (auto& layer : pool_layers) layer.set_thread_pool(pool);
        }
    };

    // RAII guard for inference: temporarily forces eval mode (so BN uses
    // running stats and never updates them), and restores the prior per-layer
    // training flag on scope exit (including on exception).  This makes
    // forward() / forward_batch() observably const w.r.t. BN training state.
    // Takes a const reference because HCNNConv::set_training is const-qualified
    // (the training flag is `mutable`).
    struct EvalModeGuard {
        const std::vector<HCNNConv>& layers;
        std::vector<bool> prev_training;
        explicit EvalModeGuard(const std::vector<HCNNConv>& l) : layers(l) {
            prev_training.reserve(layers.size());
            for (const auto& layer : layers) {
                prev_training.push_back(layer.is_training());
                layer.set_training(false);
            }
        }
        ~EvalModeGuard() {
            for (size_t i = 0; i < layers.size(); ++i)
                layers[i].set_training(prev_training[i]);
        }
    };

    // RAII guard to suppress per-sample running-stats EMA updates during
    // batch-parallel forward passes, and restore on scope exit (including on exception).
    struct BNStatsGuard {
        std::vector<HCNNConv>& layers;
        BNStatsGuard(std::vector<HCNNConv>& l) : layers(l) {
            for (auto& layer : layers)
                if (layer.has_batchnorm()) layer.set_skip_running_stats(true);
        }
        ~BNStatsGuard() {
            for (auto& layer : layers)
                if (layer.has_batchnorm()) layer.set_skip_running_stats(false);
        }
    };
};

} // namespace hcnn
