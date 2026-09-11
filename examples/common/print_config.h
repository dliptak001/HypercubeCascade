#pragma once

// Example-only config banners (ASCII for Windows consoles).

#include "Cascade.h"

#include <algorithm>
#include <cstdio>

namespace cascade_ex {

inline void PrintExciterConfig(const ExciterConfig& e)
{
    const size_t N = (e.dim < 8 * sizeof(size_t)) ? (size_t{1} << e.dim) : 0;
    const size_t M = (e.subcube_dim >= 1 && e.subcube_dim <= e.dim)
                         ? (size_t{1} << e.subcube_dim)
                         : 0;
    std::printf(
        "exciter: dim=%zu N=%zu subcube_dim=%zu M=%zu seed=%llu in_scale=%.6g wt_scale=%.6g\n",
        e.dim, N, e.subcube_dim, M,
        static_cast<unsigned long long>(e.seed),
        static_cast<double>(e.input_scaling),
        static_cast<double>(e.weight_scaling));
    std::fflush(stdout);
}

inline void PrintReservoirConfig(const ReservoirConfig& r,
                                 float realized_sr = -1.0f)
{
    const size_t N = (r.dim < 8 * sizeof(size_t)) ? (size_t{1} << r.dim) : 0;
    std::printf(
        "reservoir: dim=%zu N=%zu M=%zu seed=%llu SR_target=%.6g",
        r.dim, N, r.history_depth,
        static_cast<unsigned long long>(r.seed),
        static_cast<double>(r.spectral_radius));
    if (realized_sr >= 0.0f)
        std::printf(" SR_realized=%.3f", static_cast<double>(realized_sr));
    std::printf(" leak=%.6g in_scale=%.6g bias_scale=%.6g\n",
                static_cast<double>(r.leak_rate),
                static_cast<double>(r.input_scaling),
                static_cast<double>(r.bias_scaling));
    std::fflush(stdout);
}

/// Live readout knobs after Cascade construction (dim stamped).
/// Does not snapshot Weights() — that copies the whole blob.
inline void PrintReadoutConfig(const Readout& ro)
{
    const ReadoutConfig& r = ro.GetConfig();
    const int d = static_cast<int>(r.dim);
    int layers = (r.num_layers > 0) ? r.num_layers : std::min(d - 2, 2);
    layers = std::max(layers, 1);

    const char* pool_on = r.use_pooling ? "true" : "false";
    const char* pool_type =
        (r.pool_type == ReadoutPoolType::Avg) ? "avg" : "max";

    const char* act = "tanh";
    switch (r.activation)
    {
    case ReadoutActivation::TANH:       act = "tanh"; break;
    case ReadoutActivation::RELU:       act = "relu"; break;
    case ReadoutActivation::LEAKY_RELU: act = "leaky_relu"; break;
    case ReadoutActivation::NONE:       act = "none"; break;
    }

    std::printf(
        "readout: dim=%d layers=%d conv_channels=%d use_pooling=%s pool_type=%s "
        "activation=%s epochs=%d batch=%d lr_max=%.6g\n",
        d, layers, r.conv_channels, pool_on, pool_type, act, r.epochs,
        r.batch_size, static_cast<double>(r.lr_max));
    std::fflush(stdout);
}

inline void PrintCascadeHeader(const char* demo_name, const Cascade& cas)
{
    const CascadeConfig& cfg = cas.config();
    std::printf("%s: N=%zu dim=%zu T=%zu interstage_scale=%.6g "
                "readout_scale=%.6g ic_seed=%llu "
                "collect_threads=%zu%s\n",
                demo_name, cas.N(), cas.Dim(), cfg.T,
                static_cast<double>(cfg.interstage_scale),
                static_cast<double>(cfg.readout_scale),
                static_cast<unsigned long long>(cfg.ic_seed),
                cfg.collect_threads,
                cfg.collect_threads == 0
                    ? " (auto: leave 1-2 cores free)"
                    : "");
    PrintExciterConfig(cfg.exciter);
    PrintReservoirConfig(cfg.reservoir,
                         cas.reservoir().GetRealizedSpectralRadius());
    PrintReadoutConfig(cas.readout());
    std::fflush(stdout);
}

} // namespace cascade_ex
