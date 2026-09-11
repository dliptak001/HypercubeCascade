/// @file cascade_synth.cpp
/// @brief Synthetic multi-class fields → Cascade → held-out test.
///
/// Six classes of length-N cube fields: multi-tone carriers in the low half,
/// sparse peaks in the high half, plus deterministic noise. Not a vision claim.
///
/// Product path: field → etalon transit → interstage gain → reservoir orbit
/// → readout → held-out Accuracy.
///
/// No data files. Soft floor is on the Cascade test accuracy.

#include "Cascade.h"
#include "print_config.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <span>
#include <stdexcept>
#include <vector>

// =============================================================================
// Demo task
// =============================================================================

static constexpr int kNumClasses = 6;
static constexpr int kTrainPerClass = 64;
static constexpr int kTestPerClass = 32;
static constexpr int kTestRepBase = 10'000; // test reps never reuse train (label, rep)
static constexpr double kMinTestAcc = 0.70;
static constexpr float kNoiseStd = 0.22f;

// =============================================================================
// Product knobs (edit here)
// =============================================================================

static CascadeConfig MakeBaseConfig()
{
    CascadeConfig cfg;

    cfg.dim = 7;
    cfg.T = 50;
    cfg.interstage_scale = 5.5f;
    cfg.readout_scale = 1.0f;
    cfg.ic_seed = 1;
    cfg.collect_threads = 1;

    cfg.exciter.seed = 3458567978345987ull;
    cfg.exciter.subcube_dim = 5;
    cfg.exciter.input_scaling = 1.0f;
    cfg.exciter.weight_scaling = 0.15f;

    cfg.reservoir.spectral_radius = 0.9;
    cfg.reservoir.history_depth = 4;
    cfg.reservoir.leak_rate = 1.0;

    cfg.readout.epochs = 300;
    cfg.readout.num_outputs = kNumClasses;
    cfg.readout.task = ReadoutTask::Classification;
    cfg.readout.activation = ReadoutActivation::NONE;
    cfg.readout.batch_size = 48;
    cfg.readout.conv_channels = 4;
    cfg.readout.channel_growth = 1;
    cfg.readout.num_layers = 1;
    cfg.readout.use_pooling = false;
    cfg.readout.lr_max = 0.003f;
    cfg.readout.lr_min_frac = 0.04f;
    cfg.readout.restore_best_epoch = true;

    return cfg;
}

// =============================================================================
// Patterns
// =============================================================================

static uint32_t Mix32(uint32_t x)
{
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

static float DetNoise(int label, int rep, size_t i, float amp)
{
    const uint32_t h = Mix32(static_cast<uint32_t>(label) * 0x9E3779B9u
        ^ static_cast<uint32_t>(rep) * 0x85EBCA6Bu
        ^ static_cast<uint32_t>(i));
    const float u = (static_cast<float>(h & 0xFFFFFFu) / static_cast<float>(0xFFFFFFu)) * 2.0f
        - 1.0f;
    return amp * u;
}

/// Class-conditioned field. Shared slow ramp so DC/ramp alone is not a cue.
static void FillPattern(std::span<float> x, int label, int rep)
{
    if (label < 0 || label >= kNumClasses)
        throw std::invalid_argument("FillPattern: label out of range");
    if (x.empty())
        throw std::invalid_argument("FillPattern: empty field");

    const size_t n = x.size();
    for (size_t i = 0; i < n; ++i)
        x[i] = -1.0f;

    const size_t half = n / 2;
    constexpr float kPi = 3.14159265358979323846f;

    const float f0 = 0.55f + 0.35f * static_cast<float>(label);
    const float f1 = 1.10f + 0.40f * static_cast<float>((label * 3) % kNumClasses);
    const float phase = 0.17f * static_cast<float>(rep % 11);

    for (size_t i = 0; i < half; ++i)
    {
        const float t = static_cast<float>(i) / static_cast<float>(half);
        float v = 0.55f * std::sin(2.0f * kPi * f0 * t + phase);
        v += 0.35f * std::sin(2.0f * kPi * f1 * t - 0.5f * phase);
        v += 0.15f * (2.0f * t - 1.0f);
        v += DetNoise(label, rep, i, kNoiseStd);
        if (v > 1.0f)
            v = 1.0f;
        if (v < -1.0f)
            v = -1.0f;
        x[i] = v;
    }

    const size_t n_peaks = 3 + static_cast<size_t>(label % 3);
    for (size_t p = 0; p < n_peaks; ++p)
    {
        const size_t idx =
            half + ((p * 17u + static_cast<size_t>(label) * 13u + static_cast<size_t>(rep) * 3u)
                % (n - half));
        x[idx] = (p % 2u == 0) ? 0.9f : -0.85f;
        x[idx] += DetNoise(label, rep, idx, 0.05f);
        if (x[idx] > 1.0f)
            x[idx] = 1.0f;
        if (x[idx] < -1.0f)
            x[idx] = -1.0f;
    }
}

static void FillSet(std::vector<float>& fields, std::vector<int>& labels,
                    size_t n, int per_class, int rep_base)
{
    const size_t count = static_cast<size_t>(kNumClasses) * static_cast<size_t>(per_class);
    fields.assign(count * n, 0.0f);
    labels.assign(count, 0);
    for (size_t i = 0; i < count; ++i)
    {
        const int c = static_cast<int>(i / static_cast<size_t>(per_class));
        const int r = rep_base + static_cast<int>(i % static_cast<size_t>(per_class));
        labels[i] = c;
        FillPattern(std::span<float>(fields.data() + i * n, n), c, r);
    }
}

struct PathResult
{
    const char* name = "";
    double train_acc = 0.0;
    double test_acc = 0.0;
    double secs_collect_train = 0.0;
    double secs_test = 0.0;
};

static PathResult RunPath(const char* name, const CascadeConfig& cfg,
                          std::span<const float> train_fields,
                          std::span<const int> train_labels,
                          std::span<const float> test_fields,
                          std::span<const int> test_labels)
{
    PathResult r;
    r.name = name;

    Cascade cas(cfg);
    cascade_ex::PrintCascadeHeader(name, cas);

    auto t0 = std::chrono::steady_clock::now();
    std::printf("%s: collecting %zu fields...\n", name, train_labels.size());
    std::fflush(stdout);
    cas.CollectBatch(train_fields, train_labels);
    std::printf("%s: training readout on %zu samples...\n", name, cas.NumCollected());
    std::fflush(stdout);
    cas.TrainOnCollected();
    r.train_acc = cas.AccuracyOnCollected();
    auto t1 = std::chrono::steady_clock::now();

    r.test_acc = cas.Accuracy(test_fields, test_labels);
    auto t2 = std::chrono::steady_clock::now();

    r.secs_collect_train = std::chrono::duration<double>(t1 - t0).count();
    r.secs_test = std::chrono::duration<double>(t2 - t1).count();

    std::printf("%s: train_acc=%.3f test_acc=%.3f  time %.2f+%.2f=%.2fs "
                "(collect+train|test|total)\n",
                name, r.train_acc, r.test_acc,
                r.secs_collect_train, r.secs_test,
                r.secs_collect_train + r.secs_test);
    std::fflush(stdout);
    return r;
}

// =============================================================================

int main()
{
    int exit_code = 1;
    try
    {
        const CascadeConfig base = MakeBaseConfig();
        if (base.readout.num_outputs != kNumClasses)
            throw std::logic_error("readout.num_outputs must match kNumClasses");

        const size_t n = size_t{1} << base.dim;
        std::vector<float> train_fields;
        std::vector<int> train_labels;
        std::vector<float> test_fields;
        std::vector<int> test_labels;
        FillSet(train_fields, train_labels, n, kTrainPerClass, 0);
        FillSet(test_fields, test_labels, n, kTestPerClass, kTestRepBase);

        std::printf("cascade_synth: classes=%d train=%d/class test=%d/class "
                    "field_noise=%.2f N=%zu\n",
                    kNumClasses, kTrainPerClass, kTestPerClass,
                    static_cast<double>(kNoiseStd), n);
        std::fflush(stdout);

        const PathResult run = RunPath(
            "cascade_synth", base,
            train_fields, train_labels, test_fields, test_labels);

        if (run.test_acc < kMinTestAcc)
        {
            std::fprintf(stderr,
                         "cascade_synth: test accuracy too low "
                         "(need >= %.2f)\n",
                         kMinTestAcc);
        }
        else
        {
            exit_code = 0;
        }
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "cascade_synth: %s\n", e.what());
    }
    return exit_code;
}
