#pragma once

// Minimal MNIST IDX loader for examples (not core).
// Pixels → float in [-1, 1]; labels → int class indices 0..9.
// Enforces 28x28 images.

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace cascade_ex {

struct MnistSample
{
    std::vector<float> pixels; // 784, row-major, [-1, 1]
    int label = 0;
};

struct MnistSet
{
    std::vector<MnistSample> samples;
    [[nodiscard]] size_t size() const { return samples.size(); }
};

inline uint32_t ReadBe32(std::ifstream& f)
{
    uint8_t b[4];
    f.read(reinterpret_cast<char*>(b), 4);
    if (!f)
        throw std::runtime_error("MNIST IDX: unexpected EOF in header");
    return (uint32_t(b[0]) << 24) | (uint32_t(b[1]) << 16) | (uint32_t(b[2]) << 8)
           | uint32_t(b[3]);
}

/// Load IDX images+labels. @p max_samples 0 = all (file order, not balanced).
/// Requires 28x28, labels in [0,9].
inline MnistSet LoadMnist(const std::string& images_path,
                          const std::string& labels_path,
                          size_t max_samples = 0)
{
    std::ifstream img(images_path, std::ios::binary);
    if (!img)
        throw std::runtime_error("Cannot open MNIST images: " + images_path);

    if (ReadBe32(img) != 0x00000803u)
        throw std::runtime_error("Invalid MNIST image magic");
    const uint32_t num_images = ReadBe32(img);
    const uint32_t rows = ReadBe32(img);
    const uint32_t cols = ReadBe32(img);
    if (rows != 28 || cols != 28)
        throw std::runtime_error(
            "MNIST images must be 28x28 (got " + std::to_string(rows) + "x"
            + std::to_string(cols) + ")");
    const int pixels = 28 * 28;

    std::ifstream lbl(labels_path, std::ios::binary);
    if (!lbl)
        throw std::runtime_error("Cannot open MNIST labels: " + labels_path);
    if (ReadBe32(lbl) != 0x00000801u)
        throw std::runtime_error("Invalid MNIST label magic");
    const uint32_t num_labels = ReadBe32(lbl);
    if (num_labels != num_images)
        throw std::runtime_error("MNIST image/label count mismatch");

    size_t count = num_images;
    if (max_samples > 0 && max_samples < count)
        count = max_samples;

    MnistSet ds;
    ds.samples.resize(count);
    std::vector<uint8_t> buf(static_cast<size_t>(pixels));

    for (size_t i = 0; i < count; ++i)
    {
        img.read(reinterpret_cast<char*>(buf.data()), pixels);
        if (!img)
            throw std::runtime_error("Truncated MNIST images at " + std::to_string(i));
        auto& s = ds.samples[i];
        s.pixels.resize(static_cast<size_t>(pixels));
        for (int p = 0; p < pixels; ++p)
        {
            float v = static_cast<float>(buf[static_cast<size_t>(p)]) / 127.5f - 1.0f;
            if (v > 1.0f)
                v = 1.0f;
            if (v < -1.0f)
                v = -1.0f;
            s.pixels[static_cast<size_t>(p)] = v;
        }
        uint8_t lab = 0;
        lbl.read(reinterpret_cast<char*>(&lab), 1);
        if (!lbl)
            throw std::runtime_error("Truncated MNIST labels at " + std::to_string(i));
        if (lab > 9)
            throw std::runtime_error(
                "MNIST label out of range [0,9] at sample " + std::to_string(i)
                + " (got " + std::to_string(static_cast<int>(lab)) + ")");
        s.label = static_cast<int>(lab);
    }
    return ds;
}

} // namespace cascade_ex
