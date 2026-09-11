// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 David Liptak

#pragma once

#include "HCNNInput.h"

#include <cstddef>

namespace hcnn {

/**
 * @brief How to map a 2D single-channel image onto a length-N hypercube input
 *        with N = 2^dim and pattern length P <= N.
 *
 * Optional SDK helper for spatial → hypercube packing. Does not run the
 * network. Pair with HCNNSpatialAugmenter (2D only) and HCNN train/infer.
 *
 * Pipeline (typical):
 * @code
 *   HxW image  --(optional aug)-->  HxW
 *              --HCNNSpatialEmbedder-->  length N = 2^dim
 *              --HCNN TrainEpoch(input_length = N)-->
 * @endcode
 *
 * **Caller contract:** after embed(), pass **input_length = capacity()** (= N)
 * into HCNN. Do not pass a short pattern_length and rely on HCNN::Embed to
 * pad: network Embed always zero-pads the tail, which would **overwrite** a
 * non-zero pad_value for unused vertices.
 *
 * Layout is always a flat float[N] for **one** input channel (vertex-major).
 * Multi-channel networks are out of scope; use custom packing for c_in > 1.
 *
 * Bilinear OOB samples use pad_value (set to image background for digit-like
 * data, e.g. -1). Resize modes force a **square** SxS (aspect ratio not
 * preserved).
 */
enum class HCNNSpatialEmbedMode {
    /**
     * Write row-major HxW into out[0 .. H*W). Fill out[H*W .. N) with pad_value.
     * Requires H*W <= N. No resize. plane_side is ignored.
     * (Formerly RowMajorPad.)
     */
    PadLow,

    /**
     * Full native image in low addresses plus a centered crop in the tail:
     *   out[0 .. H*W)              = image (row-major)
     *   out[H*W .. H*W + crop_h*crop_w) = centered crop (row-major)
     *   out[H*W + crop_h*crop_w .. N)   = pad_value
     * Requires H*W <= N. Crop is the largest near-square rectangle with area
     * <= (N - H*W) that fits in HxW (floor-centered). plane_side is ignored.
     * MNIST 28x28 @ dim=10 (N=1024): crop 15x16 at (6,6), full occupancy.
     */
    PadLowCenter,

    /**
     * Bilinear-resize the image to an SxS square with S = floor(sqrt(N))
     * (or plane_side if set), write row-major into out[0 .. S*S), pad the rest.
     * Always fits: S*S <= N. Non-square inputs are distorted to square.
     */
    ResizeToFit,

    /**
     * Bilinear-resize to SxS with S = floor(sqrt(N/2)) so 2*S*S <= N
     * (or plane_side if set).
     * out[0 .. S*S)       = resized intensity (ink)
     * out[S*S .. 2*S*S)   = per-image max-normalized |grad| of that plane,
     *                       approximately [-1, 1] (blank plane -> pad_value)
     * out[2*S*S .. N)     = pad_value
     * Full occupancy when 2*S*S == N (e.g. N=512 -> S=16, N=2048 -> S=32).
     * Ink plane is not range-clipped; only |grad| is max-normalized.
     */
    DualPlaneResize,
};

/**
 * @brief Configuration for spatial -> length-N embedding.
 */
struct HCNNSpatialEmbedConfig {
    /// Hypercube dimension. Capacity N = 2^dim. Must be in [1, 30].
    int dim = 10;

    HCNNSpatialEmbedMode mode = HCNNSpatialEmbedMode::PadLow;

    /// Fill value for unused vertices, bilinear OOB, and blank dual-plane grads.
    /// For MNIST-like [-1,1] ink, use -1 (background), not the default 0.
    float pad_value = 0.0f;

    /**
     * Optional override for target square side S (ResizeToFit / DualPlaneResize).
     * 0 = automatic: floor(sqrt(N)) or floor(sqrt(N/2)) respectively.
     * If non-zero, must satisfy S*S <= N or 2*S*S <= N for the mode.
     * Ignored for PadLow / PadLowCenter.
     */
    int plane_side = 0;

    /// Vertex capacity N = 2^dim.
    int capacity() const;

    /// Validate dim / plane_side / mode; throws std::runtime_error if invalid.
    void validate() const;
};

/**
 * @brief Planned layout after embedding (for logging and buffer sizing).
 */
struct HCNNSpatialEmbedPlan {
    int dim = 0;
    int N = 0;              ///< capacity (= 2^dim), output length of embed()
    int height_in = 0;
    int width_in = 0;
    int plane_side = 0;     ///< S used for resize modes; 0 for PadLow / PadLowCenter
    int pattern_length = 0; ///< occupied floats before pad
    int crop_h = 0;         ///< PadLowCenter crop height (else 0)
    int crop_w = 0;         ///< PadLowCenter crop width (else 0)
    int crop_row0 = 0;      ///< PadLowCenter crop origin row (else 0)
    int crop_col0 = 0;      ///< PadLowCenter crop origin col (else 0)
    HCNNSpatialEmbedMode mode = HCNNSpatialEmbedMode::PadLow;
};

/**
 * @class HCNNSpatialEmbedder
 * @brief Maps 2D images into a length-N vertex buffer for HCNN (P <= N).
 *
 * Stateless aside from config. Thread-safe for concurrent embed() with a
 * fixed config.
 *
 * @code
 * hcnn::HCNNSpatialEmbedConfig cfg;
 * cfg.dim = 10;
 * cfg.mode = hcnn::HCNNSpatialEmbedMode::PadLowCenter;
 * cfg.pad_value = -1.0f;
 *
 * hcnn::HCNNSpatialEmbedder emb(cfg);
 * std::vector<float> out(emb.capacity());
 * emb.embed(img28, 28, 28, out.data());
 * // Train with input_length = emb.capacity() (== N), not pattern_length alone
 * @endcode
 */
class HCNNSpatialEmbedder {
public:
    explicit HCNNSpatialEmbedder(HCNNSpatialEmbedConfig cfg = {});

    void set_config(const HCNNSpatialEmbedConfig& cfg);
    const HCNNSpatialEmbedConfig& config() const { return cfg_; }

    /// N = 2^dim.
    int capacity() const;

    /**
     * Describe the layout that embed() will produce for this input size.
     * Throws if PadLow/PadLowCenter and H*W > N, or if config is invalid.
     */
    HCNNSpatialEmbedPlan plan(int height, int width) const;

    /**
     * Embed one row-major image into out[0 .. N). Always fully written.
     *
     * @param in      Source, length height*width
     * @param height  Rows (>= 1)
     * @param width   Cols (>= 1)
     * @param out     Destination, length capacity() (= N)
     */
    void embed(const float* in, int height, int width, float* out) const;

    /**
     * Embed `batch` images. Each sample is height*width; each output plane is N.
     * out layout: sample-major, stride = N.
     * batch == 0 is a no-op (null in/out allowed).
     */
    void embed_batch(const float* in, int batch, int height, int width,
                     float* out) const;

    /// Largest S with S*S <= N.
    static int max_square_side(int N);

    /// Largest S with 2*S*S <= N.
    static int max_dual_plane_side(int N);

private:
    HCNNSpatialEmbedConfig cfg_;

    int resolve_plane_side(int N) const;
};

/**
 * Embed a batch of HxW images into a full-capacity HCNNInputBatch
 * (`capacity() == N = 2^dim`).  Prefer this over manual embed + short length
 * so Train/Predict typed overloads cannot zero-pad over spatial pad_value.
 */
inline HCNNInputBatch pack_spatial_batch(const HCNNSpatialEmbedder& emb,
                                         const float* images, int batch,
                                         int height, int width) {
    HCNNInputBatch out;
    out.reset(batch, emb.capacity());
    if (batch > 0)
        emb.embed_batch(images, batch, height, width, out.data());
    return out;
}

/// Single image → full-capacity batch with count == 1.
inline HCNNInputBatch pack_spatial(const HCNNSpatialEmbedder& emb,
                                   const float* image, int height, int width) {
    return pack_spatial_batch(emb, image, /*batch=*/1, height, width);
}

} // namespace hcnn
