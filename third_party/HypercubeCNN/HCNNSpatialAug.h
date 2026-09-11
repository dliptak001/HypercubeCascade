// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 David Liptak

#pragma once

#include <cstdint>
#include <random>

namespace hcnn {

/**
 * @brief Configuration for optional 2D spatial augmentation.
 *
 * Operates on single-channel row-major images of any height x width.
 * Independent of hypercube DIM / vertex count -- callers map augmented
 * patterns onto the network input separately (see HCNNSpatialEmbed).
 *
 * Geometry pipeline when active:
 *   1) Optional **single inverse bilinear affine**: scale, shear, rotate,
 *      integer shift about the image center (one sample, not chained warps).
 *   2) Optional **mild elastic** (Simard-style): smooth random displacement
 *      field, then bilinear sample. Applied after affine when both run.
 *   3) Optional Gaussian **noise** after geometry; values are clipped to
 *      [value_min, value_max] when noise runs (geometry-only paths do not clip).
 *
 * Defaults are identity (no augmentation). `None()` sets `enabled = false`
 * (master off; apply is a pure copy and draws no RNG).
 *
 * Elastic cost is O(H·W·⌈3σ⌉) per displacement field (two fields); it usually
 * dominates aug time when enabled. Prefer shear-only A/B before turning elastic
 * on for MNIST (see `examples/mnist_train.cpp`).
 *
 * Keep **aug-then-embed**; do not aug on packed vertices.
 */
struct HCNNSpatialAugConfig {
    /// Uniform rotation in degrees over [-|rot_deg_max|, +|rot_deg_max|]. 0 = off.
    float rot_deg_max = 0.0f;

    /// Uniform scale factor over [scale_min, scale_max] (order-independent).
    /// Both 1 = off. Non-positive bounds are clamped to a tiny positive floor.
    float scale_min = 1.0f;
    float scale_max = 1.0f;

    /// Integer pixel translation: dy, dx ~ U{-|s|,...,+|s|} with s = |shift_max|.
    /// 0 = off.
    int shift_max = 0;

    /// Horizontal shear amount ~ U[-|shear_x_max|, +|shear_x_max|].
    /// Applied in the affine warp as x' = x + shear_x * y (about center). 0 = off.
    float shear_x_max = 0.0f;

    /// Vertical shear amount ~ U[-|shear_y_max|, +|shear_y_max|].
    /// y' = y + shear_y * x. 0 = off. MNIST slant is mostly horizontal.
    /// Require |shear_x_max| * |shear_y_max| < 0.95 so the shear matrix stays
    /// comfortably invertible (det = 1 - sx*sy).
    float shear_y_max = 0.0f;

    /// Elastic displacement scale in **pixels** (max |component| after normalize).
    /// 0 = off. Requires elastic_sigma in [kElasticSigmaMin, kElasticSigmaMax].
    float elastic_alpha = 0.0f;

    /// Gaussian smooth sigma (pixels) for the elastic displacement field.
    /// Larger = smoother / more global warp. Ignored when elastic_alpha == 0.
    float elastic_sigma = 0.0f;

    /// Additive Gaussian noise N(0, noise_sigma^2) after warp. 0 = off.
    float noise_sigma = 0.0f;

    /// Clip range after noise. Must satisfy value_min <= value_max (validated).
    float value_min = -1.0f;
    float value_max =  1.0f;

    /// Sampled value for bilinear out-of-bounds lookups.
    float border_value = 0.0f;

    /// Master switch. false => apply() copies in->out (no RNG draws).
    bool enabled = true;

    /// Min/max elastic_sigma when elastic_alpha > 0 (inclusive).
    static constexpr float kElasticSigmaMin = 0.25f;
    static constexpr float kElasticSigmaMax = 32.0f;

    /// True when apply() is a pure copy under this config (no RNG).
    bool is_identity() const;

    /// Validate field ranges; throws std::runtime_error if invalid.
    void validate() const;

    /// Master off: enabled = false (apply is memcpy, no RNG).
    static HCNNSpatialAugConfig None();
};

/**
 * @class HCNNSpatialAugmenter
 * @brief Config-owned 2D spatial augmenter (per-thread scratch for elastic).
 *
 * Thread-safe for concurrent apply() calls with distinct rng instances
 * when config is not mutated. Not safe to call set_config concurrently
 * with apply. Elastic path uses thread_local scratch (safe per thread;
 * buffers grow to the max H×W seen on that thread and do not shrink).
 *
 * @code
 * hcnn::HCNNSpatialAugConfig cfg;
 * cfg.rot_deg_max = 12.0f;
 * cfg.scale_min = 0.9f;
 * cfg.scale_max = 1.1f;
 * cfg.shift_max = 2;
 * cfg.shear_x_max = 0.15f;
 * // cfg.elastic_alpha = 1.0f;  // optional; enable after shear A/B
 * // cfg.elastic_sigma = 5.0f;
 * cfg.noise_sigma = 0.03f;
 * cfg.border_value = -1.0f;
 *
 * hcnn::HCNNSpatialAugmenter aug(cfg);
 * std::mt19937 rng(seed);
 * aug.apply(src, dst, height, width, rng);
 * @endcode
 */
class HCNNSpatialAugmenter {
public:
    /// Constructs and validates `cfg` (throws if invalid).
    explicit HCNNSpatialAugmenter(HCNNSpatialAugConfig cfg = {});

    /// Replaces config after validate() (throws if invalid).
    void set_config(const HCNNSpatialAugConfig& cfg);
    const HCNNSpatialAugConfig& config() const { return cfg_; }

    /**
     * Augment one row-major image.
     *
     * @param in      Source, length height*width. May equal out only when
     *                the config is identity or noise-only (no geometry).
     *                Affine or elastic warp requires in != out.
     * @param out     Destination, length height*width.
     * @param height  Image rows (>= 1).
     * @param width   Image cols (>= 1).
     * @param rng     Caller-owned RNG; advanced when not identity.
     */
    void apply(const float* in, float* out,
               int height, int width,
               std::mt19937& rng) const;

    /**
     * Augment `batch` images packed contiguously (sample stride = height*width).
     * Each sample draws independent geometry/noise from `rng`.
     */
    void apply_batch(const float* in, float* out,
                     int batch, int height, int width,
                     std::mt19937& rng) const;

private:
    HCNNSpatialAugConfig cfg_;
};

} // namespace hcnn
