#pragma once

#include "BaselineExtractor.h"
#include "RamanDataset.h"
#include "RamanNorm.h"

#include <cmath>
#include <span>
#include <stdexcept>
#include <vector>

/// Per-spectrum MSE of denormalized pred vs raw label.
/// Split RMSE = sqrt(mean of these).
inline double RamanPatternMse(std::span<const float> pred,
                              std::span<const float> label)
{
    if (pred.size() != label.size() || pred.empty())
    {
        throw std::invalid_argument(
            "RamanPatternMse: pred and label must be the same non-empty size");
    }
    double acc = 0.0;
    for (size_t j = 0; j < pred.size(); ++j)
    {
        const double e = static_cast<double>(label[j])
                         - static_cast<double>(pred[j]);
        acc += e * e;
    }
    return acc / static_cast<double>(pred.size());
}

inline double RamanRmse(BaselineExtractor& ex, const RamanSplit& split)
{
    if (split.count == 0)
        throw std::invalid_argument("RamanRmse: empty split");
    if (ex.N() != kN)
        throw std::invalid_argument("RamanRmse: extractor N must equal kN");

    std::vector<float> pred(ex.N());
    double sum_mse = 0.0;
    for (size_t i = 0; i < split.count; ++i)
    {
        ex.Predict(split.Spectrum(i), pred);
        sum_mse += RamanPatternMse(pred, split.Baseline(i));
    }
    return std::sqrt(sum_mse / static_cast<double>(split.count));
}

/// Train-set RMS on the already-collected features (no second Cascade map).
/// Same denorm + RMSE as @ref RamanRmse.
inline double RamanRmseOnCollected(const BaselineExtractor& ex,
                                   const RamanSplit& split)
{
    if (split.count == 0)
        throw std::invalid_argument("RamanRmseOnCollected: empty split");
    if (ex.N() != kN || ex.NumOutputs() != kN)
    {
        throw std::invalid_argument(
            "RamanRmseOnCollected: extractor N / outputs must equal kN");
    }
    if (ex.cascade().NumCollected() != split.count)
    {
        throw std::invalid_argument(
            "RamanRmseOnCollected: collected count must match split");
    }

    const auto feats = ex.cascade().CollectedFeatures();
    if (feats.size() != split.count * kN)
    {
        throw std::invalid_argument(
            "RamanRmseOnCollected: feature buffer size mismatch");
    }

    std::vector<float> yn(kN);
    std::vector<float> pred(kN);
    double sum_mse = 0.0;
    for (size_t i = 0; i < split.count; ++i)
    {
        ex.cascade().readout().PredictRaw(feats.data() + i * kN, yn.data());
        const auto nrm = RamanNorm::FromSpectrum(split.Spectrum(i));
        nrm.Invert(yn, pred);
        sum_mse += RamanPatternMse(pred, split.Baseline(i));
    }
    return std::sqrt(sum_mse / static_cast<double>(split.count));
}
