// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 David Liptak

/**
 * @file HCNNConv.h
 * @brief Hypercube convolutional layer — sparse-vertex convolution on a
 *        binary hypercube using fixed XOR masks instead of spatial grids.
 *
 * An HCNNConv layer maps c_in input channels defined on the vertices of a
 * DIM-dimensional binary hypercube (N = 2^DIM vertices) to c_out output
 * channels on the same hypercube.  For each output vertex v, the layer
 * computes:
 *
 *   out_co(v) = b_co
 *             + sum over (ci)     of w[co,ci,SELF] * in[ci, v]
 *             + sum over (ci, k)  of w[co,ci,k]    * in[ci, v ^ (1 << k)]
 *
 * where k ranges over [0, DIM) (Hamming-distance-1 neighbors) and SELF = DIM
 * is the center / self tap (mask 0).  Kernel width K = DIM + 1.
 *
 * Each tap has its own learned weight, shared across all vertices (CNN-style
 * weight sharing).  The self tap is the hypercube analogue of the center
 * weight in a spatial 3x3 kernel; neighbors are the bit-axis directions.
 *
 * All neighbor geometry is bitwise — lookup uses XOR with single-bit masks;
 * there are no adjacency lists or spatial padding.
 *
 * Memory layout is **channel-major**: element [c*N + v] stores channel c,
 * vertex v.
 */

#pragma once

#include "HCNNTypes.h"

#include <vector>
#include <random>

namespace hcnn {

class ThreadPool;

/**
 * @class HCNNConv
 * @brief One hypercube convolutional layer — private implementation.
 *
 * Not installed.  Owned by `HCNNNetwork` / `HCNN`.  Application code must not
 * include this header; in-tree tests may.  New user-facing knobs go on `HCNN`.
 *
 * Maps c_in → c_out channels on a DIM-cube with K = DIM + 1 taps (self +
 * one-bit XOR neighbors).  Layout: channel-major `data[c * N + v]`.
 */
class HCNNConv {
public:
    /**
     * @brief Construct a hypercube convolutional layer.
     *
     * Uses K = DIM + 1 taps: self (index DIM) plus DIM nearest-neighbor XOR
     * directions (indices 0 .. DIM-1).  Kernel and bias weights are
     * initialized to zero; call randomize_weights() before training.
     *
     * Requires 3 <= dim <= 30 (N = 2^dim fits in signed 32-bit int) and
     * c_in >= 1, c_out >= 1.
     *
     * @param dim            Hypercube dimension.  The layer operates on N = 2^dim vertices.
     * @param c_in           Number of input channels.
     * @param c_out          Number of output channels (filters).
     * @param activation     Activation function (default: RELU).
     * @param use_bias       If true, add a learnable per-output-channel bias (default: true).
     * @param use_batchnorm  If true, apply batch normalization between conv and activation.
     */
    HCNNConv(int dim, int c_in, int c_out,
             Activation activation = Activation::RELU,
             bool use_bias = true, bool use_batchnorm = false);

    /**
     * @brief Initialize kernel weights.
     *
     * When scale > 0, uses uniform random values in [-scale, +scale].
     * When scale <= 0, auto-selects based on activation and depth:
     *   ReLU/LeakyReLU with c_in > 1: He/Kaiming uniform, s = sqrt(6 / fan_in).
     *   Otherwise (NONE, TANH, or first layer with c_in=1):
     *     Xavier/Glorot uniform, s = sqrt(6 / (fan_in + fan_out)).
     * fan_in = c_in * K, fan_out = c_out * K  (K = DIM + 1, self + neighbors).
     *
     * Biases are reset to zero.  Momentum velocity buffers are cleared.
     *
     * @param scale  Half-width of the uniform range, or <= 0 for auto init.
     * @param rng    Mersenne Twister PRNG instance (caller-owned).
     */
    void randomize_weights(float scale, std::mt19937& rng);

    /**
     * @brief Execute the forward pass over all output channels.
     *
     * For each output channel and each vertex, accumulates the self tap and
     * DIM Hamming-1 neighbors (XOR masks), multiplies by the corresponding
     * kernel weights, adds bias, and applies the activation function.
     *
     * When batch normalization is enabled, normalization is applied between
     * the weighted sum and activation.  In training mode, per-sample statistics
     * are used and running statistics are updated.  In eval mode, running
     * statistics are used.
     *
     * @param[in]  in       Input activations, channel-major [c_in * N].
     * @param[out] out      Output activations, channel-major [c_out * N].
     * @param[out] pre_act  If non-null, receives the pre-activation values
     *                      [c_out * N].  Required by backward().
     * @param[out] bn_save  If non-null and BN enabled, receives layout
     *                      [inv_std(c_out), mean(c_out), var(c_out)] —
     *                      length get_bn_save_size().  Required for backward
     *                      / compute_gradients when BN is enabled.
     */
    void forward(const float* in, float* out, float* pre_act = nullptr,
                 float* bn_save = nullptr) const;

    /**
     * @brief Backward pass: input gradients + in-place optimizer step
     *        (SGD+momentum or Adam, per set_optimizer).
     *
     * Applies the chain rule through the activation (and BN if enabled), then:
     *   -# Computes grad_in (if non-null) using the same XOR/self structure
     *      as forward (XOR is self-inverse).
     *   -# Updates kernel / bias / BN params via the configured optimizer.
     *      Weight decay applies to kernels only (not bias or BN affine).
     *
     * When use_batchnorm, bn_save from the matching forward is required.
     *
     * @param[in]  grad_out      Gradient of loss w.r.t. output activations [c_out * N].
     * @param[in]  in            Input activations from the forward pass [c_in * N].
     * @param[in]  pre_act       Pre-activation values from the forward pass [c_out * N].
     * @param[out] grad_in       Gradient of loss w.r.t. input activations [c_in * N],
     *                           or nullptr if not needed (e.g. first layer).
     * @param      learning_rate Learning rate (eta).
     * @param      momentum      SGD momentum coefficient (mu); ignored by Adam.
     * @param      weight_decay  L2 / decoupled decay on kernels; default 0.
     * @param[in]  bn_save       BN cache from forward (required if BN enabled).
     * @param      timestep      Adam bias-correction step (t >= 1); ignored by SGD.
     * @param[in]  post_act      Optional post-activation [c_out * N].  When set
     *                           and activation==TANH, derivative uses 1 - y^2.
     */
    void backward(const float* grad_out, const float* in, const float* pre_act,
                  float* grad_in, float learning_rate, float momentum = 0.0f,
                  float weight_decay = 0.0f, const float* bn_save = nullptr,
                  int timestep = 0, const float* post_act = nullptr);

    /**
     * @brief Compute gradients without applying an optimizer update.
     *
     * Same gradient math as backward(), but writes raw grads into caller
     * buffers.  Used by mini-batch training (accumulate then apply_gradients)
     * and by numerical gradient checks.
     *
     * When use_batchnorm, bn_save from the matching forward is required.
     *
     * @param[in]  grad_out    Gradient of loss w.r.t. output activations [c_out * N].
     * @param[in]  in          Input activations from the forward pass [c_in * N].
     * @param[in]  pre_act     Pre-activation values from the forward pass [c_out * N].
     * @param[out] grad_in     Gradient of loss w.r.t. input activations [c_in * N],
     *                         or nullptr if not needed.
     * @param[out] kernel_grad Gradient of loss w.r.t. kernel weights [c_out * c_in * K].
     * @param[out] bias_grad   Gradient of loss w.r.t. bias [c_out],
     *                         or nullptr if bias is disabled.
     * @param      work_buf    Optional scratch [c_out * N]; heap fallback if null.
     * @param[in]  bn_save     BN cache from forward (required if BN enabled).
     * @param[out] bn_gamma_grad / bn_beta_grad  BN affine grads, or null.
     * @param[in]  post_act    Optional post-activation for TANH 1-y^2 path.
     */
    void compute_gradients(const float* grad_out, const float* in, const float* pre_act,
                           float* grad_in, float* kernel_grad, float* bias_grad,
                           float* work_buf = nullptr, const float* bn_save = nullptr,
                           float* bn_gamma_grad = nullptr,
                           float* bn_beta_grad = nullptr,
                           const float* post_act = nullptr) const;

    /**
     * @brief Apply externally computed (e.g. batch-averaged) gradients.
     *
     * Used by mini-batch training after compute_gradients + reduce.
     * Optimizer is SGD+momentum or Adam (same formulas as backward).
     *
     * @param kernel_grad  Averaged kernel gradients [c_out * c_in * K].
     * @param bias_grad    Averaged bias gradients [c_out], or nullptr if no bias.
     * @param learning_rate Learning rate.
     * @param momentum      SGD momentum; ignored by Adam.
     * @param weight_decay  Kernel decay; default 0.
     * @param bn_gamma_grad / bn_beta_grad  BN affine grads when BN enabled.
     * @param timestep      Adam bias-correction step (t >= 1).
     */
    void apply_gradients(const float* kernel_grad, const float* bias_grad,
                         float learning_rate, float momentum, float weight_decay = 0.0f,
                         const float* bn_gamma_grad = nullptr,
                         const float* bn_beta_grad = nullptr,
                         int timestep = 0);

    /** @name Accessors */
    ///@{
    int get_dim() const { return DIM; }       ///< Hypercube dimension.
    int get_N() const { return N; }           ///< Vertex count (2^DIM).
    int get_c_in() const { return c_in; }     ///< Number of input channels.
    int get_c_out() const { return c_out; }   ///< Number of output channels.
    int get_K() const { return K; }           ///< Kernel taps (= DIM + 1: neighbors + self).
    /// Index of the self/center tap in the last axis of the kernel (always DIM).
    int get_self_index() const { return DIM; }
    ///@}

    /// Non-owning pool for DIM>=12 paths; nullptr = single-threaded.
    /// Pool must outlive any forward/backward that uses it.
    void set_thread_pool(ThreadPool* pool) { thread_pool = pool; }

    /// Set training mode (true) or eval mode (false) for batch normalization.
    void set_training(bool training) const { training_ = training; }

    /// Current training-mode flag (for RAII save/restore in inference paths).
    bool is_training() const { return training_; }

    /// Skip running-stats EMA updates in forward() (for batch-parallel mode).
    void set_skip_running_stats(bool skip) const { skip_running_stats_ = skip; }

    /// Configure the optimizer. Allocates second-moment buffers for Adam.
    void set_optimizer(OptimizerType type, float beta1 = 0.9f,
                       float beta2 = 0.999f, float eps = 1e-8f);

    /// Whether this layer has batch normalization enabled.
    bool has_batchnorm() const { return use_batchnorm; }

    /// Size of the bn_save buffer needed by forward/backward.
    /// Layout: [inv_std(c_out), mean(c_out), var(c_out)] — 3*c_out if BN, else 0.
    /// backward/compute_gradients only read inv_std (first c_out).
    int get_bn_save_size() const { return use_batchnorm ? 3 * c_out : 0; }

    /// Size of the BN gamma/beta gradient buffers (c_out if BN, else 0).
    int get_bn_grad_size() const { return use_batchnorm ? c_out : 0; }

    /// Length of each BN affine / running-stats vector (c_out if BN, else 0).
    int get_bn_param_size() const { return use_batchnorm ? c_out : 0; }

    /// Update running mean/var from externally computed batch statistics.
    /// Applies Bessel's correction (N/(N-1)) to var before EMA update.
    void update_running_stats(const float* mean, const float* var);

    /// Zero first/second moments (SGD velocity / Adam m,v) including BN affine.
    /// Does not change optimizer type or weights.
    void clear_optimizer_moments();

    /** @name Raw weight access (for serialization and gradient checking) */
    ///@{
    float* get_kernel_data() { return kernel.data(); }
    const float* get_kernel_data() const { return kernel.data(); }
    int get_kernel_size() const { return static_cast<int>(kernel.size()); }
    float* get_bias_data() { return bias.data(); }
    const float* get_bias_data() const { return bias.data(); }
    int get_bias_size() const { return static_cast<int>(bias.size()); }
    float* get_bn_gamma_data() { return bn_gamma.data(); }
    const float* get_bn_gamma_data() const { return bn_gamma.data(); }
    float* get_bn_beta_data() { return bn_beta.data(); }
    const float* get_bn_beta_data() const { return bn_beta.data(); }
    float* get_bn_running_mean_data() { return bn_running_mean.data(); }
    const float* get_bn_running_mean_data() const { return bn_running_mean.data(); }
    float* get_bn_running_var_data() { return bn_running_var.data(); }
    const float* get_bn_running_var_data() const { return bn_running_var.data(); }
    ///@}

private:
    int DIM;          ///< Hypercube dimension.
    int N;            ///< Number of vertices, always 2^DIM.
    int c_in;         ///< Input channel count.
    int c_out;        ///< Output channel count (number of filters).
    int K;            ///< Number of kernel taps (= DIM + 1: bit-flips 0..DIM-1 + self at DIM).
    Activation activation;  ///< Activation function applied after convolution.
    bool use_bias;       ///< Whether a learnable bias term is added per output channel.
    bool use_batchnorm;  ///< Whether batch normalization is applied between conv and activation.
    mutable bool training_ = true; ///< Training mode (true) or eval mode (false) for BN.
    mutable bool skip_running_stats_ = false; ///< When true, forward() skips EMA updates (batch-parallel mode).

    std::vector<float> kernel;          ///< Kernel weights, layout [c_out * c_in * K]; k in [0,DIM) = bit k, k==DIM = self.
    std::vector<float> bias;            ///< Per-output-channel bias, size c_out (empty if bias disabled).
    std::vector<float> kernel_m;        ///< First moment (SGD velocity / Adam m) for kernel.
    std::vector<float> bias_m;          ///< First moment for bias.
    std::vector<float> kernel_m2;       ///< Second moment (Adam only) for kernel.
    std::vector<float> bias_m2;         ///< Second moment (Adam only) for bias.

    // Batch normalization parameters (empty if BN disabled)
    std::vector<float> bn_gamma;          ///< BN scale parameter [c_out].
    std::vector<float> bn_beta;           ///< BN shift parameter [c_out].
    mutable std::vector<float> bn_running_mean; ///< BN running mean [c_out] (mutable: updated in const forward).
    mutable std::vector<float> bn_running_var;  ///< BN running variance [c_out] (mutable: updated in const forward).
    std::vector<float> bn_gamma_m;        ///< First moment for BN gamma [c_out].
    std::vector<float> bn_beta_m;         ///< First moment for BN beta [c_out].
    std::vector<float> bn_gamma_m2;       ///< Second moment (Adam only) for BN gamma [c_out].
    std::vector<float> bn_beta_m2;        ///< Second moment (Adam only) for BN beta [c_out].
    static constexpr float bn_momentum_ = 0.1f;  ///< EMA momentum for running stats.
    static constexpr float bn_eps_ = 1e-5f;      ///< Epsilon for numerical stability.

    // Optimizer configuration
    OptimizerType optimizer_type_ = OptimizerType::ADAM;
    float adam_beta1_ = 0.9f, adam_beta2_ = 0.999f, adam_eps_ = 1e-8f;

    std::vector<float> backward_work_;  ///< Persistent scratch for backward() [c_out * N], grown on demand.

    ThreadPool* thread_pool = nullptr;  ///< Optional thread pool for parallel execution.

    /**
     * @brief Compute the flat index into the kernel array.
     * @param co Output channel index.
     * @param ci Input channel index.
     * @param k  Tap index: 0 .. DIM-1 = neighbor bit k; DIM = self.
     * @return   Index into the kernel vector.
     */
    int kernel_idx(int co, int ci, int k) const {
        return (co * c_in + ci) * K + k;
    }

    static constexpr float leaky_alpha_ = 0.01f; ///< LeakyReLU negative slope.

    float activate(float x) const;
    float activate_derivative(float x) const;
};

} // namespace hcnn
