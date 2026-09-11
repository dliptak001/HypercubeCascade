// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 David Liptak

#include "HCNNSpatialAug.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <string>
#include <vector>

namespace hcnn {

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

bool HCNNSpatialAugConfig::is_identity() const {
    if (!enabled) return true;
    if (std::fabs(rot_deg_max) > 0.0f) return false;
    if (std::abs(shift_max) > 0) return false;
    if (std::fabs(shear_x_max) > 0.0f) return false;
    if (std::fabs(shear_y_max) > 0.0f) return false;
    if (elastic_alpha > 0.0f) return false;
    if (noise_sigma > 0.0f) return false;
    const float s_lo = std::min(scale_min, scale_max);
    const float s_hi = std::max(scale_min, scale_max);
    return s_lo == 1.0f && s_hi == 1.0f;
}

void HCNNSpatialAugConfig::validate() const {
    auto require_finite = [](float v, const char* name) {
        if (!std::isfinite(v)) {
            throw std::runtime_error(
                std::string("HCNNSpatialAugConfig: ") + name + " must be finite");
        }
    };
    require_finite(rot_deg_max, "rot_deg_max");
    require_finite(scale_min, "scale_min");
    require_finite(scale_max, "scale_max");
    require_finite(shear_x_max, "shear_x_max");
    require_finite(shear_y_max, "shear_y_max");
    require_finite(elastic_alpha, "elastic_alpha");
    require_finite(elastic_sigma, "elastic_sigma");
    require_finite(noise_sigma, "noise_sigma");
    require_finite(value_min, "value_min");
    require_finite(value_max, "value_max");
    require_finite(border_value, "border_value");

    if (!(value_min <= value_max)) {
        throw std::runtime_error(
            "HCNNSpatialAugConfig: value_min must be <= value_max");
    }
    if (noise_sigma < 0.0f) {
        throw std::runtime_error(
            "HCNNSpatialAugConfig: noise_sigma must be >= 0");
    }
    if (elastic_alpha < 0.0f) {
        throw std::runtime_error(
            "HCNNSpatialAugConfig: elastic_alpha must be >= 0");
    }
    if (elastic_alpha > 0.0f) {
        if (elastic_sigma < kElasticSigmaMin
            || elastic_sigma > kElasticSigmaMax) {
            throw std::runtime_error(
                "HCNNSpatialAugConfig: elastic_sigma must be in "
                "[kElasticSigmaMin, kElasticSigmaMax] when elastic_alpha > 0");
        }
    }
    // Keep det = 1 - sx*sy comfortably away from 0 for any sample in the
    // box |sx|<=|sx_max|, |sy|<=|sy_max| (worst case product of maxima).
    const float sxm = std::fabs(shear_x_max);
    const float sym = std::fabs(shear_y_max);
    if (sxm * sym >= 0.95f) {
        throw std::runtime_error(
            "HCNNSpatialAugConfig: |shear_x_max|*|shear_y_max| must be < 0.95 "
            "(shear matrix near-singular)");
    }
    // rot / shift / shear maxima: magnitude is used; negative is abs.
}

HCNNSpatialAugConfig HCNNSpatialAugConfig::None() {
    HCNNSpatialAugConfig c;
    c.enabled = false;
    return c;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static float clampf(float v, float lo, float hi) {
    return std::max(lo, std::min(hi, v));
}

static int clampi(int v, int lo, int hi) {
    return std::max(lo, std::min(hi, v));
}

static float sample_bilinear(const float* img, int height, int width,
                             float y, float x, float border) {
    const int y0 = static_cast<int>(std::floor(y));
    const int x0 = static_cast<int>(std::floor(x));
    const int y1 = y0 + 1;
    const int x1 = x0 + 1;
    const float wy = y - static_cast<float>(y0);
    const float wx = x - static_cast<float>(x0);

    auto at = [img, height, width, border](int yy, int xx) -> float {
        if (yy < 0 || xx < 0 || yy >= height || xx >= width)
            return border;
        return img[yy * width + xx];
    };

    const float v00 = at(y0, x0);
    const float v01 = at(y0, x1);
    const float v10 = at(y1, x0);
    const float v11 = at(y1, x1);
    const float v0 = v00 * (1.0f - wx) + v01 * wx;
    const float v1 = v10 * (1.0f - wx) + v11 * wx;
    return v0 * (1.0f - wy) + v1 * wy;
}

/**
 * Inverse of (about center): scale -> shear -> rotate -> integer shift.
 *
 * Forward on centered coords:
 *   [x1] = s [xs]
 *   [y1]       [ys]
 *   [x2] = [1  sx] [x1]
 *   [y2]   [sy  1] [y1]
 *   then R(theta), then + (cx+dx, cy+dy).
 */
static void warp_affine(const float* src, float* dst,
                        int height, int width,
                        float deg, float scale,
                        float shear_x, float shear_y,
                        int dy, int dx,
                        float border) {
    const float cy = 0.5f * static_cast<float>(height - 1);
    const float cx = 0.5f * static_cast<float>(width - 1);
    const float s = (scale > 1e-6f) ? scale : 1.0f;
    const float rad = deg * (static_cast<float>(std::numbers::pi) / 180.0f);
    // Inverse rotation R(-theta)
    const float c = std::cos(-rad);
    const float sn = std::sin(-rad);
    const float inv_s = 1.0f / s;
    const float fdy = static_cast<float>(dy);
    const float fdx = static_cast<float>(dx);

    // Inverse shear of [1 sx; sy 1]: (1/det)[1 -sx; -sy 1]
    // Config validation keeps |sx*sy| bounded; still guard det for safety.
    float det = 1.0f - shear_x * shear_y;
    if (std::fabs(det) < 1e-6f)
        det = (det >= 0.0f) ? 1e-6f : -1e-6f;
    const float inv_det = 1.0f / det;

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            // Undo shift
            const float yy = static_cast<float>(y) - fdy - cy;
            const float xx = static_cast<float>(x) - fdx - cx;
            // Undo rotate
            const float xr = c * xx - sn * yy;
            const float yr = sn * xx + c * yy;
            // Undo shear
            const float x1 = (xr - shear_x * yr) * inv_det;
            const float y1 = (yr - shear_y * xr) * inv_det;
            // Undo scale
            const float sx = x1 * inv_s + cx;
            const float sy = y1 * inv_s + cy;
            dst[y * width + x] =
                sample_bilinear(src, height, width, sy, sx, border);
        }
    }
}

/// Separable Gaussian blur; edge clamp. `tmp` length >= H*W.
/// Kernel is cached in thread_local by (sigma, radius) to avoid per-call heap.
static void gaussian_blur_2d(float* field, int height, int width,
                             float sigma, float* tmp) {
    const int radius = std::max(1, static_cast<int>(std::ceil(3.0f * sigma)));
    const int klen = 2 * radius + 1;

    // Stack path for typical MNIST/recipe sizes (sigma <= ~21 → radius 64).
    static constexpr int kStackMaxKlen = 129; // radius <= 64
    float stack_kern[kStackMaxKlen];
    static thread_local std::vector<float> tl_kern;
    static thread_local float tl_kern_sigma =
        std::numeric_limits<float>::quiet_NaN();
    static thread_local int tl_kern_radius = -1;

    float* kern = nullptr;
    if (klen <= kStackMaxKlen) {
        kern = stack_kern;
        // Rebuild stack kernel every call (tiny; avoids TLS for common path).
        float ksum = 0.0f;
        for (int i = -radius; i <= radius; ++i) {
            const float t = static_cast<float>(i) / sigma;
            const float v = std::exp(-0.5f * t * t);
            stack_kern[i + radius] = v;
            ksum += v;
        }
        for (int i = 0; i < klen; ++i)
            stack_kern[i] /= ksum;
    } else {
        if (tl_kern_radius != radius || tl_kern_sigma != sigma) {
            tl_kern.resize(static_cast<std::size_t>(klen));
            float ksum = 0.0f;
            for (int i = -radius; i <= radius; ++i) {
                const float t = static_cast<float>(i) / sigma;
                const float v = std::exp(-0.5f * t * t);
                tl_kern[static_cast<std::size_t>(i + radius)] = v;
                ksum += v;
            }
            for (float& v : tl_kern) v /= ksum;
            tl_kern_sigma = sigma;
            tl_kern_radius = radius;
        }
        kern = tl_kern.data();
    }

    // Horizontal: field -> tmp
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float acc = 0.0f;
            for (int i = -radius; i <= radius; ++i) {
                const int xx = clampi(x + i, 0, width - 1);
                acc += kern[i + radius] * field[y * width + xx];
            }
            tmp[y * width + x] = acc;
        }
    }
    // Vertical: tmp -> field
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float acc = 0.0f;
            for (int i = -radius; i <= radius; ++i) {
                const int yy = clampi(y + i, 0, height - 1);
                acc += kern[i + radius] * tmp[yy * width + x];
            }
            field[y * width + x] = acc;
        }
    }
}

/**
 * Simard-style elastic: random N(0,1) fields, Gaussian smooth, rescale so
 * max |component| is `alpha` pixels, then bilinear sample src -> dst.
 *
 * After normalize, max(|dx[i]|, |dy[i]|) over all i equals `alpha` (or 0 if
 * the field was degenerate), so sampled source offsets are bounded by alpha
 * in each axis (plus bilinear footprint).
 */
static void warp_elastic(const float* src, float* dst,
                         int height, int width,
                         float alpha, float sigma, float border,
                         std::mt19937& rng,
                         float* dx, float* dy, float* blur_tmp) {
    const std::size_t n =
        static_cast<std::size_t>(height) * static_cast<std::size_t>(width);
    std::normal_distribution<float> unit(0.0f, 1.0f);
    for (std::size_t i = 0; i < n; ++i) {
        dx[i] = unit(rng);
        dy[i] = unit(rng);
    }
    gaussian_blur_2d(dx, height, width, sigma, blur_tmp);
    gaussian_blur_2d(dy, height, width, sigma, blur_tmp);

    float max_abs = 0.0f;
    for (std::size_t i = 0; i < n; ++i) {
        max_abs = std::max(max_abs, std::fabs(dx[i]));
        max_abs = std::max(max_abs, std::fabs(dy[i]));
    }
    const float scale = (max_abs > 1e-8f) ? (alpha / max_abs) : 0.0f;
    for (std::size_t i = 0; i < n; ++i) {
        dx[i] *= scale;
        dy[i] *= scale;
    }

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const int idx = y * width + x;
            const float sy = static_cast<float>(y) + dy[idx];
            const float sx = static_cast<float>(x) + dx[idx];
            dst[idx] = sample_bilinear(src, height, width, sy, sx, border);
        }
    }
}

// ---------------------------------------------------------------------------
// HCNNSpatialAugmenter
// ---------------------------------------------------------------------------

HCNNSpatialAugmenter::HCNNSpatialAugmenter(HCNNSpatialAugConfig cfg)
    : cfg_(cfg) {
    cfg_.validate();
}

void HCNNSpatialAugmenter::set_config(const HCNNSpatialAugConfig& cfg) {
    cfg.validate();
    cfg_ = cfg;
}

void HCNNSpatialAugmenter::apply(const float* in, float* out,
                                 int height, int width,
                                 std::mt19937& rng) const {
    if (!in || !out) {
        throw std::runtime_error("HCNNSpatialAugmenter::apply: null buffer");
    }
    if (height < 1 || width < 1) {
        throw std::runtime_error(
            "HCNNSpatialAugmenter::apply: height and width must be >= 1");
    }

    const std::size_t n =
        static_cast<std::size_t>(height) * static_cast<std::size_t>(width);

    if (!cfg_.enabled || cfg_.is_identity()) {
        if (in != out)
            std::memcpy(out, in, n * sizeof(float));
        return;
    }

    const float rot_span = std::fabs(cfg_.rot_deg_max);
    const int shift_span = std::abs(cfg_.shift_max);
    const float s_lo = std::min(cfg_.scale_min, cfg_.scale_max);
    const float s_hi = std::max(cfg_.scale_min, cfg_.scale_max);
    const float shear_x_span = std::fabs(cfg_.shear_x_max);
    const float shear_y_span = std::fabs(cfg_.shear_y_max);
    const bool do_rot = rot_span > 0.0f;
    const bool do_scale = !(s_lo == 1.0f && s_hi == 1.0f);
    const bool do_shift = shift_span > 0;
    const bool do_shear = shear_x_span > 0.0f || shear_y_span > 0.0f;
    const bool do_affine = do_rot || do_scale || do_shift || do_shear;
    const bool do_elastic = cfg_.elastic_alpha > 0.0f;
    const bool do_geom = do_affine || do_elastic;
    const bool do_noise = cfg_.noise_sigma > 0.0f;

    float deg = 0.0f;
    float scale = 1.0f;
    float shear_x = 0.0f;
    float shear_y = 0.0f;
    int dy = 0, dx = 0;

    if (do_rot) {
        std::uniform_real_distribution<float> dist(-rot_span, rot_span);
        deg = dist(rng);
    }
    if (do_scale) {
        const float lo = std::max(s_lo, 1e-6f);
        const float hi = std::max(s_hi, lo);
        std::uniform_real_distribution<float> dist(lo, hi);
        scale = dist(rng);
    }
    if (do_shift) {
        std::uniform_int_distribution<int> dist(-shift_span, shift_span);
        dy = dist(rng);
        dx = dist(rng);
    }
    if (shear_x_span > 0.0f) {
        std::uniform_real_distribution<float> dist(-shear_x_span, shear_x_span);
        shear_x = dist(rng);
    }
    if (shear_y_span > 0.0f) {
        std::uniform_real_distribution<float> dist(-shear_y_span, shear_y_span);
        shear_y = dist(rng);
    }

    if (do_geom && in == out) {
        throw std::runtime_error(
            "HCNNSpatialAugmenter::apply: geometric aug requires in != out");
    }

    // Scratch grows to max H*W seen on this thread (does not shrink).
    static thread_local std::vector<float> tl_mid;
    static thread_local std::vector<float> tl_dx;
    static thread_local std::vector<float> tl_dy;
    static thread_local std::vector<float> tl_blur;

    if (do_affine && do_elastic) {
        if (tl_mid.size() < n) tl_mid.resize(n);
        warp_affine(in, tl_mid.data(), height, width,
                    deg, scale, shear_x, shear_y, dy, dx, cfg_.border_value);
        if (tl_dx.size() < n) tl_dx.resize(n);
        if (tl_dy.size() < n) tl_dy.resize(n);
        if (tl_blur.size() < n) tl_blur.resize(n);
        warp_elastic(tl_mid.data(), out, height, width,
                     cfg_.elastic_alpha, cfg_.elastic_sigma, cfg_.border_value,
                     rng, tl_dx.data(), tl_dy.data(), tl_blur.data());
    } else if (do_affine) {
        warp_affine(in, out, height, width,
                    deg, scale, shear_x, shear_y, dy, dx, cfg_.border_value);
    } else if (do_elastic) {
        if (tl_dx.size() < n) tl_dx.resize(n);
        if (tl_dy.size() < n) tl_dy.resize(n);
        if (tl_blur.size() < n) tl_blur.resize(n);
        warp_elastic(in, out, height, width,
                     cfg_.elastic_alpha, cfg_.elastic_sigma, cfg_.border_value,
                     rng, tl_dx.data(), tl_dy.data(), tl_blur.data());
    } else if (in != out) {
        std::memcpy(out, in, n * sizeof(float));
    }

    if (do_noise) {
        std::normal_distribution<float> dist(0.0f, cfg_.noise_sigma);
        for (std::size_t i = 0; i < n; ++i)
            out[i] = clampf(out[i] + dist(rng), cfg_.value_min, cfg_.value_max);
    }
}

void HCNNSpatialAugmenter::apply_batch(const float* in, float* out,
                                       int batch, int height, int width,
                                       std::mt19937& rng) const {
    if (batch < 0) {
        throw std::runtime_error(
            "HCNNSpatialAugmenter::apply_batch: batch must be >= 0");
    }
    if (!in || !out) {
        throw std::runtime_error("HCNNSpatialAugmenter::apply_batch: null buffer");
    }
    const std::size_t plane =
        static_cast<std::size_t>(height) * static_cast<std::size_t>(width);
    for (int b = 0; b < batch; ++b) {
        apply(in + static_cast<std::size_t>(b) * plane,
              out + static_cast<std::size_t>(b) * plane,
              height, width, rng);
    }
}

} // namespace hcnn
