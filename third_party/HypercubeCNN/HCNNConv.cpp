// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 David Liptak

#include "HCNNConv.h"
#include "ThreadPool.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace hcnn {

// Minimum DIM at which per-layer threading kicks in.
// Below this, fork-join overhead exceeds the per-vertex work.
static constexpr int THREAD_DIM_THRESHOLD = 12;

// Vertex-loop tile size for cache locality.  Must be a power of 2 and a
// multiple of 8 (AVX-256 width).  At DIM=10 with T=64, 6 of 10 masks
// stay within-tile; the remaining 4 each touch exactly one other tile,
// keeping the working set in L1.
// Used only by the DIM>=THREAD_DIM_THRESHOLD threaded path; the single-
// threaded full-N helpers below use a block-pair pattern that doesn't
// need tiling (natural L1 locality from contiguous paired accesses).
static constexpr size_t TILE = 64;

// ---------------------------------------------------------------------------
// Full-N block-pair helpers (single-threaded hot path).
//
// The XOR-neighbor structure of the hypercube has a key property: for mask
// m = (1 << k), the pair (v, v^m) lies within a block of 2^(k+1) consecutive
// vertices, split into two halves of 2^k.  For v in the low half, v^m is v
// plus 2^k (the high half); for v in the high half, v^m is v minus 2^k.  So
// rewriting the XOR-indexed inner loop as a two-pass over a block's halves
// turns it into a pair of contiguous loads that auto-vectorize cleanly.
//
// The original tiled XOR-indexed loop defeats SIMD: in_ci[v ^ m] is not a
// contiguous load pattern as v increments, so the compiler either emits
// gather ops or falls back to scalar.  The block-pair form below issues
// only vanilla contiguous loads/stores.
//
// Used by the non-threaded path (DIM < THREAD_DIM_THRESHOLD = 12) of
// HCNNConv::forward, HCNNConv::backward, and HCNNConv::compute_gradients.
// The threaded path (DIM >= 12) still uses the tiled XOR form because it
// needs to operate on arbitrary [v_begin, v_end) sub-ranges that may not
// align to block boundaries.
// ---------------------------------------------------------------------------

namespace {

// Kernel layout (per (co, ci) row of length K = dim + 1):
//   kw[0 .. dim-1]  = Hamming-1 neighbor weights (bit-flip directions)
//   kw[dim]         = self / center weight (multiplies in[v], not a neighbor)
//
// Never form (1 << dim) as a mask — self is handled as a contiguous SAXPY.
// Max dim is 30; K_MAX = 31 would suffice; keep 33 for stack headroom.
static constexpr int K_MAX = 33;  // >= DIM_MAX + 1

// Forward accumulate:
//   out_co[v] = b
//             + sum_ci  kw[dim] * in_ci[v]
//             + sum_{ci,k<dim} kw[k] * in_ci[v ^ (1<<k)]
// `kernel_slice` points to kernel[co, 0, 0], i.e., c_in*K weights in row order.
inline void conv_accumulate_full(float* out_co, float b,
                                 const float* in, int c_in, int N, int dim,
                                 const float* kernel_slice)
{
    const int K = dim + 1;
    for (int v = 0; v < N; ++v) out_co[v] = b;
    for (int ci = 0; ci < c_in; ++ci) {
        const float* in_ci = in + ci * N;
        const float* kw   = kernel_slice + ci * K;

        // Self tap (contiguous loads — auto-vectorizes).
        const float w_self = kw[dim];
        for (int v = 0; v < N; ++v)
            out_co[v] += w_self * in_ci[v];

        // Neighbor taps via block-pair XOR rewrite.
        for (int k = 0; k < dim; ++k) {
            const float w    = kw[k];
            const int half   = 1 << k;
            const int block  = half << 1;
            for (int base = 0; base < N; base += block) {
                const float* in_lo  = in_ci   + base;
                const float* in_hi  = in_ci   + base + half;
                float*       out_lo = out_co  + base;
                float*       out_hi = out_co  + base + half;
                for (int i = 0; i < half; ++i) {
                    out_lo[i] += w * in_hi[i];
                    out_hi[i] += w * in_lo[i];
                }
            }
        }
    }
}

// Kernel-gradient accumulate for one (co, ci) pair:
//   grad_k_out[k]    = sum_v grad_pre_co[v] * in_ci[v ^ (1<<k)]   for k < dim
//   grad_k_out[dim]  = sum_v grad_pre_co[v] * in_ci[v]             self
// Full fp64 accumulation throughout — matches the original numerical
// semantics bit-for-bit (modulo summation order, which is a ~1e-15
// reassociation at fp64 precision).  The win vs. the original XOR-indexed
// loop is contiguous loads: the compiler emits VCVTPS2PD + VFMADD231PD
// at 4-wide fp64 rather than falling back to scalar/gather on the
// in_ci[v^m] load pattern.  The simple 2-update reduction below is left
// in form the compiler's auto-vectorizer recognizes as a reduction —
// manual unrolling into multiple scalar accumulators defeats that
// recognition and is a net loss (measured).
//
// Associative-math is scoped to this function only: the reduction is
// in fp64 so reordering introduces ~1e-15 error (negligible for fp32
// gradients), and the auto-vectorizer needs it to split the serial
// dependency chain into SIMD lanes.
#ifdef __GNUC__
#pragma GCC push_options
#pragma GCC optimize("associative-math")
#endif
inline void conv_kernel_grad_one(const float* grad_pre_co,
                                 const float* in_ci, int N, int dim,
                                 double* grad_k_out)
{
    for (int k = 0; k < dim; ++k) {
        const int half  = 1 << k;
        const int block = half << 1;
        double acc = 0.0;
        for (int base = 0; base < N; base += block) {
            const float* gp_lo = grad_pre_co + base;
            const float* gp_hi = grad_pre_co + base + half;
            const float* in_lo = in_ci       + base;
            const float* in_hi = in_ci       + base + half;
            for (int i = 0; i < half; ++i) {
                acc += static_cast<double>(gp_lo[i]) * in_hi[i];
                acc += static_cast<double>(gp_hi[i]) * in_lo[i];
            }
        }
        grad_k_out[k] = acc;
    }
    // Self gradient: sum_v grad_pre[v] * in[v]
    {
        double acc = 0.0;
        for (int v = 0; v < N; ++v)
            acc += static_cast<double>(grad_pre_co[v]) * in_ci[v];
        grad_k_out[dim] = acc;
    }
}
#ifdef __GNUC__
#pragma GCC pop_options
#endif

// Input-gradient accumulate:
//   gi_ci[v] = sum_co w_self[co,ci] * grad_pre_co[v]
//            + sum_{co,k<dim} w[co,ci,k] * grad_pre_co[v ^ (1<<k)]
// Used only for nl>=2 (first layer has grad_in = nullptr).  Same block-pair
// restructure as forward — the access pattern is identical, only the buffer
// roles swap.  Self term is a scaled add of grad_pre onto gi (XOR is self-
// inverse, and self has mask 0).
inline void conv_grad_in_full(float* gi_ci,
                              const float* grad_pre, int c_out, int N, int dim,
                              const float* kernel,
                              int c_in, int ci)
{
    const int K = dim + 1;
    for (int v = 0; v < N; ++v) gi_ci[v] = 0.0f;
    for (int co = 0; co < c_out; ++co) {
        const float* gp_co = grad_pre + co * N;
        // kernel layout: [(co * c_in + ci) * K + k]
        const float* kw = kernel + (co * c_in + ci) * K;

        // Self: gi[v] += w_self * gp[v]
        const float w_self = kw[dim];
        for (int v = 0; v < N; ++v)
            gi_ci[v] += w_self * gp_co[v];

        for (int k = 0; k < dim; ++k) {
            const float w    = kw[k];
            const int half   = 1 << k;
            const int block  = half << 1;
            for (int base = 0; base < N; base += block) {
                const float* gp_lo = gp_co + base;
                const float* gp_hi = gp_co + base + half;
                float*       gi_lo = gi_ci + base;
                float*       gi_hi = gi_ci + base + half;
                for (int i = 0; i < half; ++i) {
                    gi_lo[i] += w * gp_hi[i];
                    gi_hi[i] += w * gp_lo[i];
                }
            }
        }
    }
}

// Threaded / tiled path: accumulate self + neighbor taps over [v_begin, v_end).
// Writes out_co[v] starting from bias b (overwrites the range).
inline void conv_accumulate_range(float* out_co, float b,
                                  const float* in, int c_in, int N, int dim,
                                  int co, int K,
                                  const float* kernel_base,
                                  size_t v_begin, size_t v_end,
                                  size_t tile)
{
    for (size_t t = v_begin; t < v_end; t += tile) {
        size_t t_end = std::min(t + tile, v_end);
        for (size_t v = t; v < t_end; ++v)
            out_co[v] = b;
        for (int ci = 0; ci < c_in; ++ci) {
            const float* in_ci = in + ci * N;
            const float* kw = kernel_base + (co * c_in + ci) * K;
            // Self
            const float w_self = kw[dim];
            for (size_t v = t; v < t_end; ++v)
                out_co[v] += w_self * in_ci[v];
            // Neighbors
            for (int k = 0; k < dim; ++k) {
                const float w = kw[k];
                const uint32_t m = 1u << k;
                for (size_t v = t; v < t_end; ++v)
                    out_co[v] += w * in_ci[v ^ m];
            }
        }
    }
}

// Threaded input-grad over a vertex range for one input channel.
inline void conv_grad_in_range(float* gi, const float* grad_pre,
                               int c_out, int N, int dim, int c_in, int ci,
                               int K, const float* kernel,
                               size_t v_begin, size_t v_end, size_t tile)
{
    for (size_t t = v_begin; t < v_end; t += tile) {
        size_t t_end = std::min(t + tile, v_end);
        for (size_t v = t; v < t_end; ++v) gi[v] = 0.0f;
        for (int co = 0; co < c_out; ++co) {
            const float* gp = grad_pre + co * N;
            const float* kw = kernel + (co * c_in + ci) * K;
            const float w_self = kw[dim];
            for (size_t v = t; v < t_end; ++v)
                gi[v] += w_self * gp[v];
            for (int k = 0; k < dim; ++k) {
                const float w = kw[k];
                const uint32_t m = 1u << k;
                for (size_t v = t; v < t_end; ++v)
                    gi[v] += w * gp[v ^ m];
            }
        }
    }
}

}  // anonymous namespace

HCNNConv::HCNNConv(int dim, int c_in, int c_out, Activation activation,
                   bool use_bias, bool use_batchnorm)
    : DIM(dim), N(0), c_in(c_in), c_out(c_out),
      K(dim + 1),  // DIM neighbor directions + 1 self tap
      activation(activation), use_bias(use_bias), use_batchnorm(use_batchnorm) {
    // Align with HCNNNetwork: N = 2^dim must fit in signed 32-bit int.
    if (DIM < 3 || DIM > 30) {
        throw std::runtime_error("HCNNConv requires 3 <= DIM <= 30");
    }
    if (c_in < 1) {
        throw std::runtime_error("HCNNConv requires c_in >= 1");
    }
    if (c_out < 1) {
        throw std::runtime_error("HCNNConv requires c_out >= 1");
    }
    N = static_cast<int>(std::uint32_t{1} << DIM);

    const size_t k_elems = static_cast<size_t>(c_out) * static_cast<size_t>(c_in)
                         * static_cast<size_t>(K);
    kernel.assign(k_elems, 0.0f);
    bias.assign(use_bias ? static_cast<size_t>(c_out) : 0, 0.0f);
    kernel_m.assign(k_elems, 0.0f);
    bias_m.assign(use_bias ? static_cast<size_t>(c_out) : 0, 0.0f);
    bn_gamma.assign(use_batchnorm ? static_cast<size_t>(c_out) : 0, 1.0f);
    bn_beta.assign(use_batchnorm ? static_cast<size_t>(c_out) : 0, 0.0f);
    bn_running_mean.assign(use_batchnorm ? static_cast<size_t>(c_out) : 0, 0.0f);
    bn_running_var.assign(use_batchnorm ? static_cast<size_t>(c_out) : 0, 1.0f);
    bn_gamma_m.assign(use_batchnorm ? static_cast<size_t>(c_out) : 0, 0.0f);
    bn_beta_m.assign(use_batchnorm ? static_cast<size_t>(c_out) : 0, 0.0f);
}

void HCNNConv::randomize_weights(float scale, std::mt19937& rng) {
    // Auto-select initialization based on activation:
    //   ReLU/LeakyReLU: He/Kaiming uniform, scale = sqrt(6 / fan_in)
    //     (accounts for the variance-halving effect of ReLU)
    //   NONE (linear): Xavier/Glorot uniform, scale = sqrt(6 / (fan_in + fan_out))
    // fan_in = c_in * K, fan_out = c_out * K  (K = DIM + 1 includes self).
    if (scale <= 0.0f) {
        float fan_in  = static_cast<float>(c_in * K);
        float fan_out = static_cast<float>(c_out * K);
        // He/Kaiming for ReLU layers with c_in > 1 (intermediate layers).
        // First layer (c_in=1) uses Xavier — its input is raw data, not
        // post-ReLU activations, so the He variance assumption doesn't hold.
        if ((activation == Activation::RELU || activation == Activation::LEAKY_RELU)
            && c_in > 1) {
            scale = std::sqrt(6.0f / fan_in);
        } else {
            scale = std::sqrt(6.0f / (fan_in + fan_out));
        }
    }
    std::uniform_real_distribution<float> dist(-scale, scale);
    for (auto& w : kernel) w = dist(rng);
    if (use_bias) {
        for (auto& b : bias) b = 0.0f;
    }
    std::fill(kernel_m.begin(), kernel_m.end(), 0.0f);
    std::fill(bias_m.begin(), bias_m.end(), 0.0f);
    std::fill(kernel_m2.begin(), kernel_m2.end(), 0.0f);
    std::fill(bias_m2.begin(), bias_m2.end(), 0.0f);

    if (use_batchnorm) {
        std::fill(bn_gamma.begin(), bn_gamma.end(), 1.0f);
        std::fill(bn_beta.begin(), bn_beta.end(), 0.0f);
        std::fill(bn_running_mean.begin(), bn_running_mean.end(), 0.0f);
        std::fill(bn_running_var.begin(), bn_running_var.end(), 1.0f);
        std::fill(bn_gamma_m.begin(), bn_gamma_m.end(), 0.0f);
        std::fill(bn_beta_m.begin(), bn_beta_m.end(), 0.0f);
        std::fill(bn_gamma_m2.begin(), bn_gamma_m2.end(), 0.0f);
        std::fill(bn_beta_m2.begin(), bn_beta_m2.end(), 0.0f);
    }
}

void HCNNConv::set_optimizer(OptimizerType type, float beta1, float beta2, float eps) {
    optimizer_type_ = type;
    adam_beta1_ = beta1;
    adam_beta2_ = beta2;
    adam_eps_ = eps;
    if (type == OptimizerType::ADAM) {
        kernel_m2.assign(kernel.size(), 0.0f);
        bias_m2.assign(bias.size(), 0.0f);
        if (use_batchnorm) {
            bn_gamma_m2.assign(c_out, 0.0f);
            bn_beta_m2.assign(c_out, 0.0f);
        }
    } else {
        kernel_m2.clear(); kernel_m2.shrink_to_fit();
        bias_m2.clear(); bias_m2.shrink_to_fit();
        bn_gamma_m2.clear(); bn_gamma_m2.shrink_to_fit();
        bn_beta_m2.clear(); bn_beta_m2.shrink_to_fit();
    }
}

// ---------------------------------------------------------------------------
// Forward: vertex-level threading within each output channel.
// Each thread handles a contiguous vertex range — no write conflicts.
// Tiled: output tile stays in L1 for full ci*K accumulation + activation.
//
// When BN is enabled, the loop is split: tiled accumulation, then BN
// (needs global channel stats), then activation.  When BN is disabled,
// the original fused tiled accumulation+activation path is used.
// ---------------------------------------------------------------------------
void HCNNConv::forward(const float* in, float* out, float* pre_act,
                   float* bn_save) const {
    const bool use_threads = thread_pool && DIM >= THREAD_DIM_THRESHOLD;

    for (int co = 0; co < c_out; ++co) {
        float* out_co = out + co * N;
        float b = use_bias ? bias[co] : 0.0f;
        const float* kernel_slice = kernel.data() + co * c_in * K;

        // Tiled accumulation lambda — used by the threaded path (DIM>=12).
        // For sub-ranges [v_begin, v_end) that don't align to block boundaries,
        // we can't use the block-pair pattern, so keep the XOR form here
        // (self tap is a plain contiguous load).
        auto do_accumulate = [&](size_t v_begin, size_t v_end) {
            conv_accumulate_range(out_co, b, in, c_in, N, DIM, co, K,
                                  kernel.data(), v_begin, v_end, TILE);
        };

        // Full-N block-pair path (non-threaded — the hot single-threaded
        // path for nl=1 DIM 6..11 which is the primary config).
        auto do_accumulate_full = [&]() {
            conv_accumulate_full(out_co, b, in, c_in, N, DIM, kernel_slice);
        };

        if (use_batchnorm) {
            // Split path: accumulate → BN → activate

            // Phase 1: weighted sum (no activation yet)
            if (use_threads) {
                thread_pool->ForEach(static_cast<size_t>(N),
                    [&](size_t, size_t begin, size_t end) { do_accumulate(begin, end); });
            } else {
                do_accumulate_full();
            }

            // Phase 2: Batch normalization across all N vertices for this channel
            if (training_) {
                // Compute per-channel mean (double accumulator for DIM >= 14)
                double mean_d = 0.0;
                for (int v = 0; v < N; ++v) mean_d += out_co[v];
                float mean = static_cast<float>(mean_d / N);

                // Compute per-channel variance (double accumulator)
                double var_d = 0.0;
                for (int v = 0; v < N; ++v) {
                    float d = out_co[v] - mean;
                    var_d += static_cast<double>(d) * d;
                }
                float var = static_cast<float>(var_d / N);

                float inv_std = 1.0f / std::sqrt(var + bn_eps_);

                // Normalize, scale, shift
                for (int v = 0; v < N; ++v) {
                    float x_hat = (out_co[v] - mean) * inv_std;
                    out_co[v] = bn_gamma[co] * x_hat + bn_beta[co];
                }

                // Save inv_std, mean, var for backward and batch stats accumulation
                if (bn_save) {
                    bn_save[co] = inv_std;
                    bn_save[c_out + co] = mean;
                    bn_save[2 * c_out + co] = var;
                }

                // Update running stats (EMA) — skipped during batch-parallel mode
                if (!skip_running_stats_) {
                    float unbiased_var = var * static_cast<float>(N)
                                       / static_cast<float>(N - 1);
                    bn_running_mean[co] = (1.0f - bn_momentum_) * bn_running_mean[co]
                                        + bn_momentum_ * mean;
                    bn_running_var[co] = (1.0f - bn_momentum_) * bn_running_var[co]
                                       + bn_momentum_ * unbiased_var;
                }
            } else {
                // Eval mode: use running statistics
                float inv_std = 1.0f / std::sqrt(bn_running_var[co] + bn_eps_);
                float rm = bn_running_mean[co];
                for (int v = 0; v < N; ++v) {
                    float x_hat = (out_co[v] - rm) * inv_std;
                    out_co[v] = bn_gamma[co] * x_hat + bn_beta[co];
                }
            }

            // Phase 3: Activation
            if (pre_act) {
                float* pa = pre_act + co * N;
                for (int v = 0; v < N; ++v) {
                    pa[v] = out_co[v];
                    out_co[v] = activate(out_co[v]);
                }
            } else {
                for (int v = 0; v < N; ++v)
                    out_co[v] = activate(out_co[v]);
            }

        } else {
            // Non-BN path: accumulate (block-pair single-threaded, tiled
            // threaded) then activate in a separate linear pass.  The
            // two-pass form adds one read+write per vertex vs. the original
            // fused tiled version, but the block-pair accumulation phase
            // vectorizes cleanly — net win for nl=1 DIM<12 (the primary
            // config).
            auto do_vertices = [&](size_t v_begin, size_t v_end) {
                for (size_t t = v_begin; t < v_end; t += TILE) {
                    size_t t_end = std::min(t + TILE, v_end);
                    conv_accumulate_range(out_co, b, in, c_in, N, DIM, co, K,
                                          kernel.data(), t, t_end, TILE);
                    if (pre_act) {
                        float* pa = pre_act + co * N;
                        for (size_t v = t; v < t_end; ++v) {
                            pa[v] = out_co[v];
                            out_co[v] = activate(out_co[v]);
                        }
                    } else {
                        for (size_t v = t; v < t_end; ++v)
                            out_co[v] = activate(out_co[v]);
                    }
                }
            };

            if (use_threads) {
                thread_pool->ForEach(static_cast<size_t>(N),
                    [&](size_t, size_t begin, size_t end) { do_vertices(begin, end); });
            } else {
                do_accumulate_full();
                if (pre_act) {
                    float* pa = pre_act + co * N;
                    for (int v = 0; v < N; ++v) {
                        pa[v] = out_co[v];
                        out_co[v] = activate(out_co[v]);
                    }
                } else {
                    for (int v = 0; v < N; ++v)
                        out_co[v] = activate(out_co[v]);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Backward: vertex-level threading for input gradients (tiled).
// Channel-level threading for weight gradients (tiled reduction).
// ---------------------------------------------------------------------------
void HCNNConv::backward(const float* grad_out, const float* in, const float* pre_act,
                    float* grad_in, float learning_rate, float momentum,
                    float weight_decay, const float* bn_save, int timestep,
                    const float* post_act) {
    if (use_batchnorm && !bn_save) {
        throw std::runtime_error(
            "HCNNConv::backward: batchnorm requires bn_save from forward");
    }
    if (optimizer_type_ == OptimizerType::ADAM && timestep <= 0) {
        throw std::runtime_error(
            "HCNNConv::backward: Adam requires timestep >= 1");
    }
    const bool use_adam = (optimizer_type_ == OptimizerType::ADAM);
    const bool use_threads = thread_pool && DIM >= THREAD_DIM_THRESHOLD;

    // Adam bias-correction denominators — constant for the entire update.
    const float bc1 = use_adam ? 1.0f - static_cast<float>(std::pow(adam_beta1_, timestep)) : 1.0f;
    const float bc2 = use_adam ? 1.0f - static_cast<float>(std::pow(adam_beta2_, timestep)) : 1.0f;

    // Pre-activation gradients (gradient through activation function).
    // Persistent scratch — grown on demand, reused across calls.
    const int grad_pre_size = c_out * N;
    if (static_cast<int>(backward_work_.size()) < grad_pre_size)
        backward_work_.resize(grad_pre_size);
    float* grad_pre = backward_work_.data();
    // For TANH, derivative is 1 - tanh(x)^2 = 1 - y^2 where y is the
    // post-activation.  When the caller supplies post_act we avoid the
    // redundant std::tanh in activate_derivative.
    if (activation == Activation::TANH && post_act != nullptr) {
        for (int i = 0; i < grad_pre_size; ++i) {
            float y = post_act[i];
            grad_pre[i] = grad_out[i] * (1.0f - y * y);
        }
    } else {
        for (int i = 0; i < grad_pre_size; ++i)
            grad_pre[i] = grad_out[i] * activate_derivative(pre_act[i]);
    }

    // BN backward: transform grad from "w.r.t. BN output" to "w.r.t. raw sum"
    if (use_batchnorm) {
        for (int co = 0; co < c_out; ++co) {
            float* gp = grad_pre + co * N;
            const float* pa = pre_act + co * N;
            float inv_std = bn_save[co];
            float gamma_co = bn_gamma[co];
            float inv_gamma = (gamma_co != 0.0f) ? (1.0f / gamma_co) : 0.0f;
            float inv_N = 1.0f / static_cast<float>(N);

            // Pass 1: compute dgamma, dbeta, and intermediate sums (double accumulators)
            double dgamma_d = 0.0, dbeta_d = 0.0;
            double sum_dx_hat_d = 0.0, sum_dx_hat_xhat_d = 0.0;
            for (int v = 0; v < N; ++v) {
                float x_hat = (pa[v] - bn_beta[co]) * inv_gamma;
                float dx_hat = gp[v] * gamma_co;
                dgamma_d += gp[v] * x_hat;
                dbeta_d += gp[v];
                sum_dx_hat_d += dx_hat;
                sum_dx_hat_xhat_d += dx_hat * x_hat;
            }

            float dgamma = static_cast<float>(dgamma_d);
            float dbeta = static_cast<float>(dbeta_d);
            float mean_dx = static_cast<float>(sum_dx_hat_d * inv_N);
            float mean_dx_xhat = static_cast<float>(sum_dx_hat_xhat_d * inv_N);

            // Pass 2: compute gradient w.r.t. raw weighted sum (replaces grad_pre)
            for (int v = 0; v < N; ++v) {
                float x_hat = (pa[v] - bn_beta[co]) * inv_gamma;
                float dx_hat = gp[v] * gamma_co;
                gp[v] = inv_std * (dx_hat - mean_dx - x_hat * mean_dx_xhat);
            }

            // Update BN parameters
            if (use_adam) {
                bn_gamma_m[co] = adam_beta1_ * bn_gamma_m[co] + (1.0f - adam_beta1_) * dgamma;
                bn_gamma_m2[co] = adam_beta2_ * bn_gamma_m2[co] + (1.0f - adam_beta2_) * dgamma * dgamma;
                float mh = bn_gamma_m[co] / bc1;
                float vh = bn_gamma_m2[co] / bc2;
                bn_gamma[co] -= learning_rate * mh / (std::sqrt(vh) + adam_eps_);
                bn_beta_m[co] = adam_beta1_ * bn_beta_m[co] + (1.0f - adam_beta1_) * dbeta;
                bn_beta_m2[co] = adam_beta2_ * bn_beta_m2[co] + (1.0f - adam_beta2_) * dbeta * dbeta;
                mh = bn_beta_m[co] / bc1;
                vh = bn_beta_m2[co] / bc2;
                bn_beta[co] -= learning_rate * mh / (std::sqrt(vh) + adam_eps_);
            } else {
                bn_gamma_m[co] = momentum * bn_gamma_m[co] + dgamma;
                bn_gamma[co] -= learning_rate * bn_gamma_m[co];
                bn_beta_m[co] = momentum * bn_beta_m[co] + dbeta;
                bn_beta[co] -= learning_rate * bn_beta_m[co];
            }
        }
    }

    // Input gradient: vertex-level parallelism, tiled (threaded) or block-pair (not).
    if (grad_in) {
        for (int ci = 0; ci < c_in; ++ci) {
            float* gi = grad_in + ci * N;

            auto do_vertices = [&](size_t v_begin, size_t v_end) {
                conv_grad_in_range(gi, grad_pre, c_out, N, DIM, c_in, ci, K,
                                   kernel.data(), v_begin, v_end, TILE);
            };

            if (use_threads) {
                thread_pool->ForEach(static_cast<size_t>(N),
                    [&](size_t, size_t b, size_t e) { do_vertices(b, e); });
            } else {
                conv_grad_in_full(gi, grad_pre, c_out, N, DIM,
                                  kernel.data(), c_in, ci);
            }
        }
    }

    // Weight update: channel-level parallelism across c_out.  Each call
    // sweeps full [0, N) via the block-pair kernel-grad helper (self + nbrs).
    auto do_weight_update = [&](int co) {
        const float* gp = grad_pre + co * N;
        double grad_k[K_MAX];  // K = DIM + 1 <= 33
        for (int ci = 0; ci < c_in; ++ci) {
            const float* in_ci = in + ci * N;
            conv_kernel_grad_one(gp, in_ci, N, DIM, grad_k);
            for (int k = 0; k < K; ++k) {
                int ki = kernel_idx(co, ci, k);
                float g = static_cast<float>(grad_k[k]);
                if (use_adam) {
                    kernel_m[ki] = adam_beta1_ * kernel_m[ki] + (1.0f - adam_beta1_) * g;
                    kernel_m2[ki] = adam_beta2_ * kernel_m2[ki] + (1.0f - adam_beta2_) * g * g;
                    float mh = kernel_m[ki] / bc1;
                    float vh = kernel_m2[ki] / bc2;
                    kernel[ki] -= learning_rate * (mh / (std::sqrt(vh) + adam_eps_) + weight_decay * kernel[ki]);
                } else {
                    g += weight_decay * kernel[ki];
                    kernel_m[ki] = momentum * kernel_m[ki] + g;
                    kernel[ki] -= learning_rate * kernel_m[ki];
                }
            }
        }
        if (use_bias) {
            double grad_b_d = 0.0;
            for (int v = 0; v < N; ++v) grad_b_d += gp[v];
            float grad_b = static_cast<float>(grad_b_d);
            if (use_adam) {
                bias_m[co] = adam_beta1_ * bias_m[co] + (1.0f - adam_beta1_) * grad_b;
                bias_m2[co] = adam_beta2_ * bias_m2[co] + (1.0f - adam_beta2_) * grad_b * grad_b;
                float mh = bias_m[co] / bc1;
                float vh = bias_m2[co] / bc2;
                bias[co] -= learning_rate * mh / (std::sqrt(vh) + adam_eps_);
            } else {
                bias_m[co] = momentum * bias_m[co] + grad_b;
                bias[co] -= learning_rate * bias_m[co];
            }
        }
    };

    if (use_threads) {
        thread_pool->ForEach(static_cast<size_t>(c_out),
            [&](size_t, size_t b, size_t e) {
                for (size_t co = b; co < e; ++co) do_weight_update(static_cast<int>(co));
            });
    } else {
        for (int co = 0; co < c_out; ++co) do_weight_update(co);
    }
}

// ---------------------------------------------------------------------------
// compute_gradients: same tiling strategy as backward, but writes raw
// gradients to caller buffers instead of updating weights.
// ---------------------------------------------------------------------------
void HCNNConv::compute_gradients(const float* grad_out, const float* in, const float* pre_act,
                             float* grad_in, float* kernel_grad, float* bias_grad,
                             float* work_buf, const float* bn_save,
                             float* bn_gamma_grad, float* bn_beta_grad,
                             const float* post_act) const {
    if (use_batchnorm && !bn_save) {
        throw std::runtime_error(
            "HCNNConv::compute_gradients: batchnorm requires bn_save from forward");
    }
    const bool use_threads = thread_pool && DIM >= THREAD_DIM_THRESHOLD;

    // work_buf must be at least c_out * N floats if provided.
    // Falls back to heap allocation if nullptr (backward compat).
    std::vector<float> grad_pre_storage;
    float* grad_pre;
    if (work_buf) {
        grad_pre = work_buf;
    } else {
        grad_pre_storage.resize(static_cast<size_t>(c_out) * static_cast<size_t>(N));
        grad_pre = grad_pre_storage.data();
    }
    // For TANH, derivative is 1 - tanh(x)^2 = 1 - y^2 where y is the
    // post-activation.  When the caller supplies post_act we avoid the
    // redundant std::tanh in activate_derivative.
    if (activation == Activation::TANH && post_act != nullptr) {
        const int grad_pre_size = c_out * N;
        for (int i = 0; i < grad_pre_size; ++i) {
            float y = post_act[i];
            grad_pre[i] = grad_out[i] * (1.0f - y * y);
        }
    } else {
        for (int i = 0; i < c_out * N; ++i)
            grad_pre[i] = grad_out[i] * activate_derivative(pre_act[i]);
    }

    // BN backward: transform grad from "w.r.t. BN output" to "w.r.t. raw sum"
    if (use_batchnorm) {
        for (int co = 0; co < c_out; ++co) {
            float* gp = grad_pre + co * N;
            const float* pa = pre_act + co * N;
            float inv_std = bn_save[co];
            float gamma_co = bn_gamma[co];
            float inv_gamma = (gamma_co != 0.0f) ? (1.0f / gamma_co) : 0.0f;
            float inv_N = 1.0f / static_cast<float>(N);

            double dgamma_d = 0.0, dbeta_d = 0.0;
            double sum_dx_hat_d = 0.0, sum_dx_hat_xhat_d = 0.0;
            for (int v = 0; v < N; ++v) {
                float x_hat = (pa[v] - bn_beta[co]) * inv_gamma;
                float dx_hat = gp[v] * gamma_co;
                dgamma_d += gp[v] * x_hat;
                dbeta_d += gp[v];
                sum_dx_hat_d += dx_hat;
                sum_dx_hat_xhat_d += dx_hat * x_hat;
            }

            float mean_dx = static_cast<float>(sum_dx_hat_d * inv_N);
            float mean_dx_xhat = static_cast<float>(sum_dx_hat_xhat_d * inv_N);

            for (int v = 0; v < N; ++v) {
                float x_hat = (pa[v] - bn_beta[co]) * inv_gamma;
                float dx_hat = gp[v] * gamma_co;
                gp[v] = inv_std * (dx_hat - mean_dx - x_hat * mean_dx_xhat);
            }

            if (bn_gamma_grad) bn_gamma_grad[co] = static_cast<float>(dgamma_d);
            if (bn_beta_grad) bn_beta_grad[co] = static_cast<float>(dbeta_d);
        }
    }

    // Input gradient: vertex-level, tiled (threaded) or full-N block-pair.
    // Dead code for nl=1 (first/only layer has grad_in=nullptr), live for nl>=2.
    if (grad_in) {
        for (int ci = 0; ci < c_in; ++ci) {
            float* gi = grad_in + ci * N;

            auto do_vertices = [&](size_t v_begin, size_t v_end) {
                conv_grad_in_range(gi, grad_pre, c_out, N, DIM, c_in, ci, K,
                                   kernel.data(), v_begin, v_end, TILE);
            };

            if (use_threads) {
                thread_pool->ForEach(static_cast<size_t>(N),
                    [&](size_t, size_t b, size_t e) { do_vertices(b, e); });
            } else {
                conv_grad_in_full(gi, grad_pre, c_out, N, DIM,
                                  kernel.data(), c_in, ci);
            }
        }
    }

    // Kernel + bias gradient: channel-level parallelism across c_out.
    // Each call sweeps full [0, N) via the block-pair helper, so threading
    // (which splits c_out) and the helper's full-N assumption are
    // compatible — no non-threaded/threaded split needed here.
    auto do_kernel_grad = [&](int co) {
        const float* gp = grad_pre + co * N;
        double grad_k_buf[K_MAX];  // K = DIM + 1 <= 33
        for (int ci = 0; ci < c_in; ++ci) {
            const float* in_ci = in + ci * N;
            conv_kernel_grad_one(gp, in_ci, N, DIM, grad_k_buf);
            for (int k = 0; k < K; ++k)
                kernel_grad[kernel_idx(co, ci, k)] =
                    static_cast<float>(grad_k_buf[k]);
        }
        if (bias_grad && use_bias) {
            double grad_b_d = 0.0;
            for (int v = 0; v < N; ++v) grad_b_d += gp[v];
            bias_grad[co] = static_cast<float>(grad_b_d);
        }
    };

    if (use_threads) {
        thread_pool->ForEach(static_cast<size_t>(c_out),
            [&](size_t, size_t b, size_t e) {
                for (size_t co = b; co < e; ++co) do_kernel_grad(static_cast<int>(co));
            });
    } else {
        for (int co = 0; co < c_out; ++co) do_kernel_grad(co);
    }
}

// ---------------------------------------------------------------------------
// apply_gradients: apply pre-computed (averaged) gradients with momentum SGD.
// ---------------------------------------------------------------------------
void HCNNConv::apply_gradients(const float* kernel_grad, const float* bias_grad,
                           float learning_rate, float momentum, float weight_decay,
                           const float* bn_gamma_grad_in, const float* bn_beta_grad_in,
                           int timestep) {
    if (optimizer_type_ == OptimizerType::ADAM && timestep <= 0) {
        throw std::runtime_error(
            "HCNNConv::apply_gradients: Adam requires timestep >= 1");
    }
    const bool use_adam = (optimizer_type_ == OptimizerType::ADAM);
    const float bc1 = use_adam ? 1.0f - static_cast<float>(std::pow(adam_beta1_, timestep)) : 1.0f;
    const float bc2 = use_adam ? 1.0f - static_cast<float>(std::pow(adam_beta2_, timestep)) : 1.0f;
    int total_k = c_out * c_in * K;

    if (use_adam) {
        for (int i = 0; i < total_k; ++i) {
            float g = kernel_grad[i];
            kernel_m[i] = adam_beta1_ * kernel_m[i] + (1.0f - adam_beta1_) * g;
            kernel_m2[i] = adam_beta2_ * kernel_m2[i] + (1.0f - adam_beta2_) * g * g;
            float mh = kernel_m[i] / bc1;
            float vh = kernel_m2[i] / bc2;
            kernel[i] -= learning_rate * (mh / (std::sqrt(vh) + adam_eps_) + weight_decay * kernel[i]);
        }
    } else {
        for (int i = 0; i < total_k; ++i) {
            float g = kernel_grad[i] + weight_decay * kernel[i];
            kernel_m[i] = momentum * kernel_m[i] + g;
            kernel[i] -= learning_rate * kernel_m[i];
        }
    }

    if (use_bias && bias_grad) {
        for (int co = 0; co < c_out; ++co) {
            if (use_adam) {
                float g = bias_grad[co];
                bias_m[co] = adam_beta1_ * bias_m[co] + (1.0f - adam_beta1_) * g;
                bias_m2[co] = adam_beta2_ * bias_m2[co] + (1.0f - adam_beta2_) * g * g;
                float mh = bias_m[co] / bc1;
                float vh = bias_m2[co] / bc2;
                bias[co] -= learning_rate * mh / (std::sqrt(vh) + adam_eps_);
            } else {
                bias_m[co] = momentum * bias_m[co] + bias_grad[co];
                bias[co] -= learning_rate * bias_m[co];
            }
        }
    }

    if (use_batchnorm && bn_gamma_grad_in && bn_beta_grad_in) {
        for (int co = 0; co < c_out; ++co) {
            if (use_adam) {
                float gg = bn_gamma_grad_in[co], bg = bn_beta_grad_in[co];
                bn_gamma_m[co] = adam_beta1_ * bn_gamma_m[co] + (1.0f - adam_beta1_) * gg;
                bn_gamma_m2[co] = adam_beta2_ * bn_gamma_m2[co] + (1.0f - adam_beta2_) * gg * gg;
                float mh = bn_gamma_m[co] / bc1;
                float vh = bn_gamma_m2[co] / bc2;
                bn_gamma[co] -= learning_rate * mh / (std::sqrt(vh) + adam_eps_);
                bn_beta_m[co] = adam_beta1_ * bn_beta_m[co] + (1.0f - adam_beta1_) * bg;
                bn_beta_m2[co] = adam_beta2_ * bn_beta_m2[co] + (1.0f - adam_beta2_) * bg * bg;
                mh = bn_beta_m[co] / bc1;
                vh = bn_beta_m2[co] / bc2;
                bn_beta[co] -= learning_rate * mh / (std::sqrt(vh) + adam_eps_);
            } else {
                bn_gamma_m[co] = momentum * bn_gamma_m[co] + bn_gamma_grad_in[co];
                bn_gamma[co] -= learning_rate * bn_gamma_m[co];
                bn_beta_m[co] = momentum * bn_beta_m[co] + bn_beta_grad_in[co];
                bn_beta[co] -= learning_rate * bn_beta_m[co];
            }
        }
    }
}

void HCNNConv::update_running_stats(const float* mean, const float* var) {
    if (!use_batchnorm || !mean || !var) return;
    for (int co = 0; co < c_out; ++co) {
        float unbiased_var = var[co] * static_cast<float>(N)
                           / static_cast<float>(N - 1);
        bn_running_mean[co] = (1.0f - bn_momentum_) * bn_running_mean[co]
                            + bn_momentum_ * mean[co];
        bn_running_var[co] = (1.0f - bn_momentum_) * bn_running_var[co]
                           + bn_momentum_ * unbiased_var;
    }
}

void HCNNConv::clear_optimizer_moments() {
    std::fill(kernel_m.begin(), kernel_m.end(), 0.0f);
    std::fill(bias_m.begin(), bias_m.end(), 0.0f);
    std::fill(kernel_m2.begin(), kernel_m2.end(), 0.0f);
    std::fill(bias_m2.begin(), bias_m2.end(), 0.0f);
    std::fill(bn_gamma_m.begin(), bn_gamma_m.end(), 0.0f);
    std::fill(bn_beta_m.begin(), bn_beta_m.end(), 0.0f);
    std::fill(bn_gamma_m2.begin(), bn_gamma_m2.end(), 0.0f);
    std::fill(bn_beta_m2.begin(), bn_beta_m2.end(), 0.0f);
}

// Compile-time switch: HCNN_FAST_TANH replaces std::tanh with a rational
// Padé/Lambert approximation.  README option 3b.  When the macro is not
// defined the build remains bit-identical to the exact-tanh baseline.
static inline float hcnn_tanh(float x) {
#ifdef HCNN_FAST_TANH
    // tanh(x) ≈ x · (27 + x²) / (27 + 9·x²).  Asymptote is x/9, not 1,
    // so clamp the output to keep saturation correct for |x| ≳ 3.
    const float x2 = x * x;
    const float y  = x * (27.0f + x2) / (27.0f + 9.0f * x2);
    return y > 1.0f ? 1.0f : (y < -1.0f ? -1.0f : y);
#else
    return std::tanh(x);
#endif
}

float HCNNConv::activate(float x) const {
    switch (activation) {
        case Activation::RELU:       return (x > 0.0f) ? x : 0.0f;
        case Activation::LEAKY_RELU: return (x > 0.0f) ? x : leaky_alpha_ * x;
        case Activation::TANH:       return hcnn_tanh(x);
        default:                     return x;
    }
}

float HCNNConv::activate_derivative(float x) const {
    switch (activation) {
        case Activation::RELU:       return (x > 0.0f) ? 1.0f : 0.0f;
        case Activation::LEAKY_RELU: return (x > 0.0f) ? 1.0f : leaky_alpha_;
        case Activation::TANH: {
            // d/dx tanh(x) = 1 - tanh(x)^2.  We receive the pre-activation
            // here (consistent with the RELU/LEAKY path that conditions on
            // pre-activation sign), so recompute tanh(x) once.  The cost is
            // one extra `tanh` per gradient element; profile-wise this is a
            // wash because the forward pass already dominates conv compute.
            float t = hcnn_tanh(x);
            return 1.0f - t * t;
        }
        default:                     return 1.0f;
    }
}

} // namespace hcnn
