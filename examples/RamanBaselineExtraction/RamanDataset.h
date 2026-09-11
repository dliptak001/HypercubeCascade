#pragma once

#include "BaselineExtractor.h"

#include <cstddef>
#include <filesystem>
#include <span>
#include <vector>

struct RamanSplit
{
    size_t count = 0;
    std::vector<float> spectra;
    std::vector<float> baselines;

    [[nodiscard]] std::span<const float> Spectrum(size_t i) const
    {
        return {spectra.data() + i * kN, kN};
    }

    [[nodiscard]] std::span<const float> Baseline(size_t i) const
    {
        return {baselines.data() + i * kN, kN};
    }
};

RamanSplit LoadRamanSplit(const std::filesystem::path& dir, size_t prefix);
RamanSplit LoadRamanIndices(const std::filesystem::path& dir,
                            std::span<const int> indices);
