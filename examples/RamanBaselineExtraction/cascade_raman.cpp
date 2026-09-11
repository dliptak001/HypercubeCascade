#include "BaselineExtractor.h"
#include "RamanDataset.h"
#include "RamanNorm.h"
#include "RamanPaths.h"
#include "RamanScore.h"
#include "print_config.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

static constexpr int kTrainSamples = 10000;
static constexpr int kTestSamples = 2000;
static constexpr bool kSkipTrain = false;
static constexpr bool kReportWalkGain = false;

static BaselineExtractor* s_ex = nullptr;
static const RamanSplit* s_train = nullptr;

static void EpochTick(int epoch, double)
{
    const double train_rmse = RamanRmseOnCollected(*s_ex, *s_train);
    std::printf("cascade_raman: epoch=%d train_rmse=%.6f\n", epoch, train_rmse);
    std::fflush(stdout);
}

// Same probe as cascade_mnist: one packed field through Cascade::Run, then
// mean |value| at each stage. ~1 is a live field, ~0 is crushed.
static void ReportStageScales(const char* name, Cascade& cas,
                              std::span<const float> packed)
{
    cas.Run(packed);
    auto mean_abs = [](std::span<const float> x) {
        if (x.empty())
            return 0.0;
        double a = 0.0;
        for (float v : x)
            a += std::fabs(static_cast<double>(v));
        return a / static_cast<double>(x.size());
    };
    std::printf("%s: stage scales after one packed train field "
                "(mean |value| over N=%zu; ~1 is a live field, "
                "~0 is crushed)\n",
                name, cas.N());
    std::printf("%s:   Exciter output (etalon transit)              "
                "mean|y|=%.4g\n",
                name, mean_abs(cas.LastExciter()));
    std::printf("%s:   Reservoir drive (Exciter * interstage_scale) "
                "mean|d|=%.4g\n",
                name, mean_abs(cas.LastInterstage()));
    std::printf("%s:   Reservoir output (live state)                "
                "mean|z|=%.4g\n",
                name, mean_abs(cas.LastReservoir()));
    std::printf("%s:   Readout features (Reservoir * readout_scale) "
                "mean|f|=%.4g\n",
                name, mean_abs(cas.LastFeatures()));
    std::fflush(stdout);
}

// Walk-IC probe: rms(y) / rms(x) after input_scaling, over N vertices.
// Uses collected cascade features (no second map). Delete with kReportWalkGain.
static void ReportWalkIcGain(const BaselineExtractor& ex, const RamanSplit& split)
{
    if (ex.cascade().NumCollected() != split.count)
    {
        throw std::logic_error(
            "walk_ic: collected count must match split");
    }

    const auto feats = ex.cascade().CollectedFeatures();
    const float scale = ex.config().exciter.input_scaling;
    const double inv_n = 1.0 / static_cast<double>(kN);

    double sum_gain = 0.0;
    double sum_rx = 0.0;
    double sum_ry = 0.0;
    double min_gain = 0.0;
    double max_gain = 0.0;
    size_t n_ok = 0;
    std::vector<float> xn(kN);

    for (size_t i = 0; i < split.count; ++i)
    {
        const auto nrm = RamanNorm::FromSpectrum(split.Spectrum(i));
        nrm.Apply(split.Spectrum(i), xn);

        double acc_x = 0.0;
        double acc_y = 0.0;
        const float* y = feats.data() + i * kN;
        for (size_t j = 0; j < kN; ++j)
        {
            const double x = static_cast<double>(xn[j]) * static_cast<double>(scale);
            const double yy = static_cast<double>(y[j]);
            acc_x += x * x;
            acc_y += yy * yy;
        }
        const double rms_x = std::sqrt(acc_x * inv_n);
        const double rms_y = std::sqrt(acc_y * inv_n);
        if (rms_x <= 0.0)
            continue;
        const double g = rms_y / rms_x;
        if (n_ok == 0)
        {
            min_gain = g;
            max_gain = g;
        }
        else
        {
            min_gain = std::min(min_gain, g);
            max_gain = std::max(max_gain, g);
        }
        sum_gain += g;
        sum_rx += rms_x;
        sum_ry += rms_y;
        ++n_ok;
    }

    if (n_ok == 0)
    {
        std::printf("cascade_raman: walk_ic n=0 (all flat)\n");
        std::fflush(stdout);
        return;
    }
    const double inv = 1.0 / static_cast<double>(n_ok);
    std::printf(
        "cascade_raman: walk_ic n=%zu mean_gain=%.6f min_gain=%.6f max_gain=%.6f "
        "mean_rms_x=%.6g mean_rms_y=%.6g\n",
        n_ok, sum_gain * inv, min_gain, max_gain, sum_rx * inv, sum_ry * inv);
    std::fflush(stdout);
}

int main()
{
    int exit_code = 1;
    try
    {
        if (kTrainSamples < 0 || kTestSamples < 0)
        {
            throw std::invalid_argument(
                "kTrainSamples / kTestSamples must be >= 0 (0 = whole split)");
        }

        const std::filesystem::path root(kRamanDataRoot);
        const auto train = LoadRamanSplit(
            root / "Training", static_cast<size_t>(kTrainSamples));
        const auto test = LoadRamanSplit(
            root / "Validation", static_cast<size_t>(kTestSamples));

        if (train.count == 0)
            throw std::runtime_error("empty training split");

        const std::string stem(kRamanModelStem);

        CascadeConfig cfg = MakeBaseConfig();
        if (!kSkipTrain)
            cfg.readout.epoch_tick = EpochTick;

        BaselineExtractor ex(cfg);
        if (ex.N() != kN)
            throw std::logic_error("extractor N must equal kN");
        s_ex = &ex;
        s_train = &train;

        std::printf("cascade_raman: train=%zu val=%zu skip_train=%s\n",
                    train.count, test.count,
                    kSkipTrain ? "true" : "false");
        cascade_ex::PrintCascadeHeader("cascade_raman", ex.cascade());
        {
            std::vector<float> xn(kN);
            const auto nrm = RamanNorm::FromSpectrum(train.Spectrum(0));
            nrm.Apply(train.Spectrum(0), xn);
            ReportStageScales("cascade_raman", ex.cascade(), xn);
        }
        {
            // Same resolve as HCNN ThreadPool: 0 = auto (hw-1 workers + caller),
            // 1 = caller only, N > 1 = N workers + caller.
            const unsigned hw = std::thread::hardware_concurrency();
            const size_t knob = ex.config().readout.num_threads;
            size_t pool_nt = 1;
            if (knob == 0)
                pool_nt = (hw > 1) ? static_cast<size_t>(hw) : 1;
            else if (knob > 1)
                pool_nt = knob + 1;
            std::printf("cascade_raman: hw=%u pool_nt=%zu\n", hw, pool_nt);
            std::fflush(stdout);
        }

        if (kSkipTrain)
        {
            ex.LoadReadout(stem);
            if (!ex.cascade().readout().IsTrained())
            {
                throw std::runtime_error(
                    "cascade_raman: LoadReadout did not mark trained");
            }
            std::printf("cascade_raman: loaded %s.hcnw\n", stem.c_str());
            std::fflush(stdout);
        }
        else
        {
            ex.Collect(train);
            std::printf("cascade_raman: collected\n");
            std::fflush(stdout);
            if (kReportWalkGain)
                ReportWalkIcGain(ex, train);

            const auto t0 = std::chrono::steady_clock::now();
            ex.Train();
            const auto t1 = std::chrono::steady_clock::now();
            const double train_secs =
                std::chrono::duration<double>(t1 - t0).count();
            std::printf("cascade_raman: trained time=%.2fs best_epoch=%d\n",
                        train_secs, ex.cascade().readout().BestEpoch());
            std::fflush(stdout);

            ex.SaveReadout(stem);
            {
                BaselineExtractor check(cfg);
                check.LoadReadout(stem);
                std::vector<float> a(kN), b(kN);
                const auto probe = train.Spectrum(0);
                ex.Predict(probe, a);
                check.Predict(probe, b);
                for (size_t i = 0; i < kN; ++i)
                {
                    if (std::fabs(a[i] - b[i]) > 1e-3f)
                    {
                        throw std::runtime_error(
                            "cascade_raman: saved readout failed reload check");
                    }
                }
            }
            std::printf("cascade_raman: saved %s.hcnw %s.arch.json\n",
                        stem.c_str(), stem.c_str());
            std::fflush(stdout);
        }

        const double train_rmse = RamanRmse(ex, train);
        const double val_rmse = RamanRmse(ex, test);
        std::printf("cascade_raman: train_rmse=%.6f val_rmse=%.6f\n",
                    train_rmse, val_rmse);
        std::fflush(stdout);
        exit_code = 0;
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "cascade_raman: %s\n", e.what());
    }
    return exit_code;
}
