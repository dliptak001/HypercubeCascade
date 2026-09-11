// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 David Liptak

#include "HCNNSpatialEmbed.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

namespace hcnn {

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

int HCNNSpatialEmbedConfig::capacity() const {
    if (dim < 1 || dim > 30) {
        throw std::runtime_error("HCNNSpatialEmbedConfig: dim must be in [1, 30]");
    }
    return 1 << dim;
}

void HCNNSpatialEmbedConfig::validate() const {
    if (dim < 1 || dim > 30) {
        throw std::runtime_error("HCNNSpatialEmbedConfig: dim must be in [1, 30]");
    }
    const int N = 1 << dim;
    if (plane_side < 0) {
        throw std::runtime_error("HCNNSpatialEmbedConfig: plane_side must be >= 0");
    }
    if (plane_side > 0) {
        const long long S = plane_side;
        if (mode == HCNNSpatialEmbedMode::ResizeToFit) {
            if (S * S > static_cast<long long>(N)) {
                throw std::runtime_error(
                    "HCNNSpatialEmbedConfig: plane_side*plane_side exceeds N=2^dim");
            }
        } else if (mode == HCNNSpatialEmbedMode::DualPlaneResize) {
            if (2 * S * S > static_cast<long long>(N)) {
                throw std::runtime_error(
                    "HCNNSpatialEmbedConfig: 2*plane_side*plane_side exceeds N=2^dim");
            }
        }
        // PadLow / PadLowCenter ignore plane_side (no error).
    }
}

// ---------------------------------------------------------------------------
// Static helpers
// ---------------------------------------------------------------------------

int HCNNSpatialEmbedder::max_square_side(int N) {
    if (N < 1) return 0;
    int s = static_cast<int>(std::floor(std::sqrt(static_cast<double>(N))));
    while (s > 0 && static_cast<long long>(s) * s > N) --s;
    return s;
}

int HCNNSpatialEmbedder::max_dual_plane_side(int N) {
    if (N < 2) return 0;
    int s = static_cast<int>(std::floor(std::sqrt(static_cast<double>(N) / 2.0)));
    while (s > 0 && 2LL * s * s > N) --s;
    return s;
}

/// Largest near-square center crop with area <= rem that fits in HxW.
/// Tie-break: min |h-w|, then prefer wider (larger w), then smaller h.
/// Origin is floor-centered: row0 = (H-h)/2, col0 = (W-w)/2.
static void choose_center_crop(int H, int W, int rem,
                               int& crop_h, int& crop_w,
                               int& row0, int& col0) {
    crop_h = crop_w = row0 = col0 = 0;
    if (rem < 1 || H < 1 || W < 1) return;

    int best_a = 0;
    int best_aspect = 0;
    bool have = false;

    for (int h = 1; h <= H; ++h) {
        const int w = std::min(W, rem / h);
        if (w < 1) continue;
        const int a = h * w;
        const int aspect = std::abs(h - w);
        const bool better =
            !have
            || a > best_a
            || (a == best_a && aspect < best_aspect)
            || (a == best_a && aspect == best_aspect && w > crop_w)
            || (a == best_a && aspect == best_aspect && w == crop_w && h < crop_h);
        if (better) {
            have = true;
            best_a = a;
            best_aspect = aspect;
            crop_h = h;
            crop_w = w;
        }
    }
    if (have) {
        row0 = (H - crop_h) / 2;
        col0 = (W - crop_w) / 2;
    }
}

// ---------------------------------------------------------------------------
// Bilinear resize + gradient magnitude
// ---------------------------------------------------------------------------

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

// Half-pixel aligned resize src (h x w) -> dst (S x S).
static void resize_to_square(const float* src, int height, int width,
                             float* dst, int S, float border) {
    if (S < 1) return;
    const float sy_scale = static_cast<float>(height) / static_cast<float>(S);
    const float sx_scale = static_cast<float>(width) / static_cast<float>(S);
    for (int y = 0; y < S; ++y) {
        for (int x = 0; x < S; ++x) {
            const float sy = (static_cast<float>(y) + 0.5f) * sy_scale - 0.5f;
            const float sx = (static_cast<float>(x) + 0.5f) * sx_scale - 0.5f;
            dst[y * S + x] = sample_bilinear(src, height, width, sy, sx, border);
        }
    }
}

// Finite-difference |grad| on SxS; per-image max-norm -> roughly [-1, 1].
// Blank / constant -> fill with pad_value.
static void grad_magnitude_plane(const float* img, float* out, int S, float pad_value) {
    float gmax = 0.0f;
    for (int y = 0; y < S; ++y) {
        for (int x = 0; x < S; ++x) {
            const int x1 = (x + 1 < S) ? x + 1 : x;
            const int y1 = (y + 1 < S) ? y + 1 : y;
            const float c  = img[y * S + x];
            const float dx = img[y * S + x1] - c;
            const float dy = img[y1 * S + x] - c;
            const float g  = std::sqrt(dx * dx + dy * dy);
            out[y * S + x] = g;
            if (g > gmax) gmax = g;
        }
    }
    const int n = S * S;
    if (gmax < 1e-8f) {
        std::fill(out, out + n, pad_value);
        return;
    }
    const float inv = 1.0f / gmax;
    for (int i = 0; i < n; ++i) {
        const float u = out[i] * inv;     // [0, 1]
        out[i] = 2.0f * u - 1.0f;         // [-1, 1]
    }
}

// ---------------------------------------------------------------------------
// Embedder
// ---------------------------------------------------------------------------

HCNNSpatialEmbedder::HCNNSpatialEmbedder(HCNNSpatialEmbedConfig cfg)
    : cfg_(cfg) {
    cfg_.validate();
}

void HCNNSpatialEmbedder::set_config(const HCNNSpatialEmbedConfig& cfg) {
    cfg.validate();
    cfg_ = cfg;
}

int HCNNSpatialEmbedder::capacity() const {
    return cfg_.capacity();
}

int HCNNSpatialEmbedder::resolve_plane_side(int N) const {
    if (cfg_.plane_side > 0)
        return cfg_.plane_side;
    if (cfg_.mode == HCNNSpatialEmbedMode::DualPlaneResize)
        return max_dual_plane_side(N);
    if (cfg_.mode == HCNNSpatialEmbedMode::ResizeToFit)
        return max_square_side(N);
    return 0;
}

HCNNSpatialEmbedPlan HCNNSpatialEmbedder::plan(int height, int width) const {
    cfg_.validate();
    if (height < 1 || width < 1) {
        throw std::runtime_error(
            "HCNNSpatialEmbedder::plan: height and width must be >= 1");
    }

    HCNNSpatialEmbedPlan p;
    p.dim = cfg_.dim;
    p.N = cfg_.capacity();
    p.height_in = height;
    p.width_in = width;
    p.mode = cfg_.mode;
    p.crop_h = p.crop_w = p.crop_row0 = p.crop_col0 = 0;

    switch (cfg_.mode) {
    case HCNNSpatialEmbedMode::PadLow: {
        const long long need =
            static_cast<long long>(height) * static_cast<long long>(width);
        if (need > p.N) {
            throw std::runtime_error(
                "HCNNSpatialEmbedder::plan: H*W=" + std::to_string(need)
                + " exceeds N=2^dim=" + std::to_string(p.N)
                + " (use ResizeToFit, DualPlaneResize, or increase dim)");
        }
        p.plane_side = 0;
        p.pattern_length = static_cast<int>(need);
        break;
    }
    case HCNNSpatialEmbedMode::PadLowCenter: {
        const long long need =
            static_cast<long long>(height) * static_cast<long long>(width);
        if (need > p.N) {
            throw std::runtime_error(
                "HCNNSpatialEmbedder::plan: H*W=" + std::to_string(need)
                + " exceeds N=2^dim=" + std::to_string(p.N)
                + " (PadLowCenter needs H*W <= N; raise dim or use a resize mode)");
        }
        const int rem = p.N - static_cast<int>(need);
        choose_center_crop(height, width, rem,
                           p.crop_h, p.crop_w, p.crop_row0, p.crop_col0);
        p.plane_side = 0;
        p.pattern_length = static_cast<int>(need) + p.crop_h * p.crop_w;
        break;
    }
    case HCNNSpatialEmbedMode::ResizeToFit: {
        p.plane_side = resolve_plane_side(p.N);
        if (p.plane_side < 1) {
            throw std::runtime_error(
                "HCNNSpatialEmbedder::plan: ResizeToFit needs N >= 1");
        }
        p.pattern_length = p.plane_side * p.plane_side;
        break;
    }
    case HCNNSpatialEmbedMode::DualPlaneResize: {
        p.plane_side = resolve_plane_side(p.N);
        if (p.plane_side < 1) {
            throw std::runtime_error(
                "HCNNSpatialEmbedder::plan: DualPlaneResize needs N >= 2");
        }
        const long long two_planes =
            2LL * static_cast<long long>(p.plane_side) * p.plane_side;
        if (two_planes > p.N) {
            throw std::runtime_error(
                "HCNNSpatialEmbedder::plan: dual plane layout exceeds N");
        }
        p.pattern_length = static_cast<int>(two_planes);
        break;
    }
    default:
        throw std::runtime_error(
            "HCNNSpatialEmbedder::plan: unknown HCNNSpatialEmbedMode");
    }
    return p;
}

void HCNNSpatialEmbedder::embed(const float* in, int height, int width,
                                float* out) const {
    if (!in || !out) {
        throw std::runtime_error("HCNNSpatialEmbedder::embed: null buffer");
    }

    const HCNNSpatialEmbedPlan p = plan(height, width);
    const int N = p.N;
    const float pad = cfg_.pad_value;

    // Default: fill entire buffer with pad, then overwrite occupied region.
    std::fill(out, out + N, pad);

    switch (cfg_.mode) {
    case HCNNSpatialEmbedMode::PadLow: {
        const std::size_t n =
            static_cast<std::size_t>(height) * static_cast<std::size_t>(width);
        std::memcpy(out, in, n * sizeof(float));
        break;
    }
    case HCNNSpatialEmbedMode::PadLowCenter: {
        const std::size_t n_pix =
            static_cast<std::size_t>(height) * static_cast<std::size_t>(width);
        std::memcpy(out, in, n_pix * sizeof(float));
        if (p.crop_h > 0 && p.crop_w > 0) {
            float* tail = out + n_pix;
            const std::size_t row_bytes =
                static_cast<std::size_t>(p.crop_w) * sizeof(float);
            for (int y = 0; y < p.crop_h; ++y) {
                const float* row = in + (p.crop_row0 + y) * width + p.crop_col0;
                std::memcpy(tail + static_cast<std::size_t>(y) * static_cast<std::size_t>(p.crop_w),
                            row, row_bytes);
            }
        }
        break;
    }
    case HCNNSpatialEmbedMode::ResizeToFit: {
        const int S = p.plane_side;
        resize_to_square(in, height, width, out, S, pad);
        break;
    }
    case HCNNSpatialEmbedMode::DualPlaneResize: {
        const int S = p.plane_side;
        const int plane = S * S;
        // Safety: grad plane must not overlap ink (plan already enforces 2*S*S <= N).
        if (static_cast<long long>(plane) * 2 > N) {
            throw std::runtime_error(
                "HCNNSpatialEmbedder::embed: dual plane overflow (internal)");
        }
        resize_to_square(in, height, width, out, S, pad);
        grad_magnitude_plane(out, out + plane, S, pad);
        break;
    }
    default:
        throw std::runtime_error(
            "HCNNSpatialEmbedder::embed: unknown HCNNSpatialEmbedMode");
    }
}

void HCNNSpatialEmbedder::embed_batch(const float* in, int batch,
                                      int height, int width,
                                      float* out) const {
    if (batch < 0) {
        throw std::runtime_error(
            "HCNNSpatialEmbedder::embed_batch: batch must be >= 0");
    }
    if (batch == 0)
        return;  // no-op; null buffers allowed
    if (!in || !out) {
        throw std::runtime_error("HCNNSpatialEmbedder::embed_batch: null buffer");
    }
    const int N = capacity();
    const std::size_t src_plane =
        static_cast<std::size_t>(height) * static_cast<std::size_t>(width);
    const std::size_t dst_stride = static_cast<std::size_t>(N);
    for (int b = 0; b < batch; ++b) {
        embed(in + static_cast<std::size_t>(b) * src_plane,
              height, width,
              out + static_cast<std::size_t>(b) * dst_stride);
    }
}

} // namespace hcnn
