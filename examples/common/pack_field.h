#pragma once

// Example-only packing helpers (caller-owned field → length-N drive).
// Not part of the Cascade core library.
//
// MNIST packs go through vendored HCNN SpatialEmbed:
//   PadLow       — full HxW in low verts + pad tail
//   PadLowCenter — full HxW + centered crop in the remaining budget
// DualPlaneResize / ResizeToFit remain available on HCNNSpatialEmbedMode
// for ablations but are not used by the MNIST demo.

#include "HCNNSpatialEmbed.h"

#include <cstddef>
#include <span>
#include <stdexcept>

namespace cascade_ex {

/// Fill @p out (length N) with @p data in low addresses; pad the rest.
/// Requires data.size() <= out.size().
inline void PackPadLow(std::span<const float> data, std::span<float> out,
                       float pad_value = -1.0f)
{
    if (data.size() > out.size())
        throw std::invalid_argument(
            "PackPadLow: data longer than field N (need pad room or larger dim)");
    for (size_t i = 0; i < data.size(); ++i)
        out[i] = data[i];
    for (size_t i = data.size(); i < out.size(); ++i)
        out[i] = pad_value;
}

/// Build an embedder for MNIST-style packing (reuse for the whole run).
inline hcnn::HCNNSpatialEmbedder MakeMnistEmbedder(int dim,
                                                   hcnn::HCNNSpatialEmbedMode mode,
                                                   float pad_value = -1.0f)
{
    if (dim < 5 || dim > 16)
        throw std::invalid_argument("MakeMnistEmbedder: dim must be in [5, 16]");
    if (mode != hcnn::HCNNSpatialEmbedMode::PadLow
        && mode != hcnn::HCNNSpatialEmbedMode::PadLowCenter
        && mode != hcnn::HCNNSpatialEmbedMode::DualPlaneResize
        && mode != hcnn::HCNNSpatialEmbedMode::ResizeToFit)
    {
        throw std::invalid_argument("MakeMnistEmbedder: unsupported embed mode");
    }
    hcnn::HCNNSpatialEmbedConfig ec;
    ec.dim = dim;
    ec.mode = mode;
    ec.pad_value = pad_value;
    ec.plane_side = 0;
    return hcnn::HCNNSpatialEmbedder(ec);
}

/// Pack one 28x28 MNIST sample via a pre-built embedder.
inline void PackMnist28(const float* img28x28,
                        const hcnn::HCNNSpatialEmbedder& emb,
                        std::span<float> out)
{
    if (img28x28 == nullptr)
        throw std::invalid_argument("PackMnist28: null image");
    if (static_cast<size_t>(emb.capacity()) != out.size())
        throw std::invalid_argument(
            "PackMnist28: out.size() must equal embedder capacity N");
    emb.embed(img28x28, 28, 28, out.data());
}

} // namespace cascade_ex
