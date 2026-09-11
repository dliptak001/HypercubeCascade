/// @file cascade_mnist.cpp
/// @brief MNIST → pack length-N field → Cascade → held-out test.
///
/// Pipeline: IDX load → file-order prefix → PadLowCenter pack → CollectBatch
/// → TrainOnCollected → Accuracy on packed test fields.
///
/// Product path: pack → etalon transit → interstage gain → reservoir orbit
/// → readout → held-out Accuracy.
/// Optional test-only AWGN on the packed field (train clean). Demo knobs
/// kTestNoiseSweep / Start / End / Step. Off = clean test. On = train once,
/// then score start, start+step, ... while <= end and print a table.
///
/// Data: C:\HypercubeCascade\data

#include "Cascade.h"
#include "find_data_dir.h"
#include "mnist_idx.h"
#include "pack_field.h"
#include "print_config.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

// =============================================================================
// Pack
// =============================================================================

enum class PackMode
{
    PadLow,
    PadLowCenter, // default: 28x28 + centered crop fills N=1024
};

static hcnn::HCNNSpatialEmbedMode ToEmbedMode(PackMode pack)
{
    switch (pack)
    {
    case PackMode::PadLow: return hcnn::HCNNSpatialEmbedMode::PadLow;
    case PackMode::PadLowCenter: return hcnn::HCNNSpatialEmbedMode::PadLowCenter;
    }
    return hcnn::HCNNSpatialEmbedMode::PadLowCenter;
}

static const char* PackModeName(PackMode pack)
{
    switch (pack)
    {
    case PackMode::PadLow: return "PadLow";
    case PackMode::PadLowCenter: return "PadLowCenter";
    }
    return "?";
}

// =============================================================================
// Product knobs (edit here)
// =============================================================================

static constexpr int kEpochs = 35;

static CascadeConfig MakeBaseConfig()
{
    CascadeConfig cfg;

    // PadLow / PadLowCenter need N >= 784 → dim >= 10.
    cfg.dim = 10;
    cfg.T = 50;
    cfg.interstage_scale = 1.0;
    cfg.readout_scale = 1.0f;
    cfg.ic_seed = 12;

    cfg.exciter.seed = 3458567978345987ull;
    cfg.exciter.subcube_dim = 5;
    cfg.exciter.input_scaling = 0.1f;
    cfg.exciter.weight_scaling = 0.15f;

    cfg.reservoir.seed = 13871537636959942979ull;
    cfg.reservoir.spectral_radius = 0.9;
    cfg.reservoir.history_depth = 2;
    cfg.reservoir.leak_rate = 0.9;
    cfg.reservoir.bias_scaling = 0.0;
    cfg.reservoir.input_scaling = 1.0;

    cfg.readout.epochs = kEpochs;
    cfg.readout.num_outputs = 10;
    cfg.readout.task = ReadoutTask::Classification;
    cfg.readout.activation = ReadoutActivation::NONE;
    cfg.readout.batch_size = 64;
    cfg.readout.conv_channels = 16;
    cfg.readout.channel_growth = 1;
    cfg.readout.num_layers = 1;
    cfg.readout.use_pooling = true;
    cfg.readout.lr_max = 0.0015f;
    cfg.readout.lr_min_frac = 0.01f;
    cfg.readout.restore_best_epoch = true;

    return cfg;
}

// =============================================================================
// Demo / task (not product config)
// =============================================================================

static constexpr PackMode kPack = PackMode::PadLowCenter;

// MNIST train is 60000, test 10000. A short demo is 1000 / 500.
static constexpr int kTrainSamples = 60000;
static constexpr int kTestSamples = 10000;
static constexpr float kPad = -1.0f;
static constexpr int kImgSide = 28;
static constexpr int kImgPixels = kImgSide * kImgSide;
static constexpr double kMinTestAcc = 0.50;

static constexpr bool kTestNoiseSweep = true;
static constexpr float kTestNoiseStart = 0.0f;
static constexpr float kTestNoiseEnd = 1.0f;
static constexpr float kTestNoiseStep = 0.1f;
static constexpr unsigned kTestNoiseSeedBase = 0x7E57u;

// =============================================================================

static void PackSet(const cascade_ex::MnistSet& ds,
                    const hcnn::HCNNSpatialEmbedder& emb,
                    std::vector<float>& fields,
                    std::vector<int>& labels)
{
    const size_t n = static_cast<size_t>(emb.capacity());
    fields.assign(ds.size() * n, 0.0f);
    labels.assign(ds.size(), 0);
    for (size_t i = 0; i < ds.size(); ++i)
    {
        labels[i] = ds.samples[i].label;
        auto row = std::span<float>(fields.data() + i * n, n);
        cascade_ex::PackMnist28(ds.samples[i].pixels.data(), emb, row);
    }
}

static std::vector<float> MakeTestNoiseGrid()
{
    std::vector<float> grid;
    if (!kTestNoiseSweep)
        return grid;
    if (!(kTestNoiseStart >= 0.0f) || !std::isfinite(kTestNoiseStart)
        || !(kTestNoiseEnd >= 0.0f) || !std::isfinite(kTestNoiseEnd)
        || !std::isfinite(kTestNoiseStep) || !(kTestNoiseStep > 0.0f))
    {
        throw std::invalid_argument(
            "cascade_mnist: test noise sweep needs finite start/end >= 0 "
            "and step > 0");
    }
    if (kTestNoiseStart > kTestNoiseEnd)
    {
        throw std::invalid_argument(
            "cascade_mnist: kTestNoiseStart must be <= kTestNoiseEnd");
    }
    if (kTestNoiseEnd <= kTestNoiseStart)
    {
        grid.push_back(kTestNoiseStart);
        return grid;
    }
    const double n = std::floor(
        (static_cast<double>(kTestNoiseEnd) - kTestNoiseStart)
        / kTestNoiseStep + 1e-6) + 1.0;
    if (n > 1000.0)
    {
        throw std::invalid_argument(
            "cascade_mnist: test noise grid would exceed 1000 points");
    }
    const int count = static_cast<int>(n);
    grid.reserve(static_cast<size_t>(count));
    for (int i = 0; i < count; ++i)
    {
        grid.push_back(kTestNoiseStart
            + kTestNoiseStep * static_cast<float>(i));
    }
    return grid;
}

static void PrintTestNoiseReport(const std::vector<float>& grid)
{
    if (!kTestNoiseSweep)
    {
        std::printf("cascade_mnist: test_noise=off\n");
        return;
    }
    std::printf(
        "cascade_mnist: test_noise=sweep start=%g end=%g step=%g "
        "(%zu points) seed_base=0x%X (train once, then each sigma)\n",
        static_cast<double>(kTestNoiseStart),
        static_cast<double>(kTestNoiseEnd),
        static_cast<double>(kTestNoiseStep),
        grid.size(), kTestNoiseSeedBase);
}

/// In-place i.i.d. Gaussian on a packed field (no clamp). No-op if σ <= 0.
static void AddTestFieldNoise(std::span<float> field, size_t sample_index,
                              float sigma)
{
    if (sigma <= 0.0f)
        return;
    std::mt19937 rng(kTestNoiseSeedBase
        + static_cast<unsigned>(sample_index) * 9973u);
    std::normal_distribution<float> dist(0.0f, sigma);
    for (float& v : field)
        v += dist(rng);
}

static double ScoreNoisyTest(Cascade& cas,
                             std::span<const float> clean_fields,
                             std::span<const int> labels,
                             float sigma)
{
    const size_t n = cas.N();
    if (sigma <= 0.0f)
        return cas.Accuracy(clean_fields, labels);

    std::vector<float> noisy(clean_fields.begin(), clean_fields.end());
    if (noisy.size() != labels.size() * n)
        throw std::logic_error("test pack size mismatch");
    for (size_t i = 0; i < labels.size(); ++i)
    {
        AddTestFieldNoise(
            std::span<float>(noisy.data() + i * n, n), i, sigma);
    }
    return cas.Accuracy(noisy, labels);
}

struct PathResult
{
    const char* name = "";
    double train_acc = 0.0;
    double test_acc = 0.0;
    double secs_collect_train = 0.0;
    double secs_test = 0.0;
};

static void TrainPath(const char* name, Cascade& cas,
                      std::span<const float> train_fields,
                      std::span<const int> train_labels,
                      PathResult& r)
{
    r.name = name;

    cascade_ex::PrintCascadeHeader(name, cas);

    if (!train_fields.empty())
    {
        cas.Run(train_fields.subspan(0, cas.N()));
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

    auto t0 = std::chrono::steady_clock::now();
    std::printf("%s: collecting %zu fields...\n", name, train_labels.size());
    std::fflush(stdout);
    cas.CollectBatch(train_fields, train_labels);

    std::printf("%s: training readout on %zu samples...\n", name, cas.NumCollected());
    std::fflush(stdout);
    cas.TrainOnCollected();
    r.train_acc = cas.AccuracyOnCollected();
    auto t1 = std::chrono::steady_clock::now();
    r.secs_collect_train = std::chrono::duration<double>(t1 - t0).count();
}

static void PrintPathSummary(const PathResult& r)
{
    std::printf("%s: train_acc=%.3f test_acc=%.3f  time %.1f+%.1f=%.1fs "
                "(collect+train|test|total)\n",
                r.name, r.train_acc, r.test_acc,
                r.secs_collect_train, r.secs_test,
                r.secs_collect_train + r.secs_test);
    std::fflush(stdout);
}

static void PrintNoiseSweepTable(const char* name, double train_acc,
                                 double collect_train_s,
                                 std::span<const float> sigmas,
                                 std::span<const double> accs,
                                 std::span<const double> secs)
{
    std::printf("%s: train_acc=%.3f  (collect+train %.1fs); "
                "test noise sweep (%zu sigmas)\n",
                name, train_acc, collect_train_s, sigmas.size());
    std::printf("  sigma      test_acc    test_s\n");
    std::printf("  ---------  --------    ------\n");
    for (size_t i = 0; i < sigmas.size(); ++i)
    {
        std::printf("  %9.4f  %8.3f    %6.1f\n",
                    static_cast<double>(sigmas[i]), accs[i], secs[i]);
    }
    std::fflush(stdout);
}

/// Score each sigma on a trained instance. Returns clean (σ=0) acc, or -1
/// if the grid has no clean point.
static double RunNoiseSweep(Cascade& cas, const char* name,
                            double train_acc, double collect_train_s,
                            std::span<const float> clean_fields,
                            std::span<const int> labels,
                            std::span<const float> grid,
                            std::vector<double>& accs,
                            std::vector<double>& secs)
{
    accs.clear();
    secs.clear();
    accs.reserve(grid.size());
    secs.reserve(grid.size());
    double clean_acc = -1.0;
    for (float sigma : grid)
    {
        auto t0 = std::chrono::steady_clock::now();
        const double acc = ScoreNoisyTest(cas, clean_fields, labels, sigma);
        auto t1 = std::chrono::steady_clock::now();
        accs.push_back(acc);
        secs.push_back(std::chrono::duration<double>(t1 - t0).count());
        if (sigma <= 0.0f)
            clean_acc = acc;
    }
    PrintNoiseSweepTable(name, train_acc, collect_train_s, grid, accs, secs);
    return clean_acc;
}

// =============================================================================

int main()
{
    int exit_code = 1;
    try
    {
        const CascadeConfig base = MakeBaseConfig();
        const size_t n_field = size_t{1} << base.dim;
        if (n_field < static_cast<size_t>(kImgPixels))
        {
            throw std::invalid_argument(
                "cascade_mnist: pack needs N >= 784 (dim >= 10), got N="
                + std::to_string(n_field));
        }

        const auto data_dir = cascade_ex::FindMnistDataDir(nullptr);
        const auto data_str =
            std::filesystem::absolute(data_dir).lexically_normal().make_preferred().string();

        std::printf("cascade_mnist: loading IDX from %s\n", data_str.c_str());
        std::fflush(stdout);

        if (kTrainSamples < 0 || kTestSamples < 0)
        {
            throw std::invalid_argument(
                "cascade_mnist: kTrainSamples / kTestSamples must be >= 0 "
                "(0 = whole file)");
        }
        const auto train = cascade_ex::LoadMnist(
            (data_dir / "train-images-idx3-ubyte").string(),
            (data_dir / "train-labels-idx1-ubyte").string(),
            static_cast<size_t>(kTrainSamples));
        const auto test = cascade_ex::LoadMnist(
            (data_dir / "t10k-images-idx3-ubyte").string(),
            (data_dir / "t10k-labels-idx1-ubyte").string(),
            static_cast<size_t>(kTestSamples));

        const auto emb = cascade_ex::MakeMnistEmbedder(
            static_cast<int>(base.dim), ToEmbedMode(kPack), kPad);
        if (static_cast<size_t>(emb.capacity()) != n_field)
            throw std::logic_error("embed capacity does not match N");

        const auto plan = emb.plan(kImgSide, kImgSide);
        std::printf("cascade_mnist: pack=%s train=%zu test=%zu epochs=%d\n",
                    PackModeName(kPack), train.size(), test.size(), kEpochs);
        if (kPack == PackMode::PadLowCenter)
        {
            std::printf(
                "cascade_mnist: PadLowCenter full=%dx%d@ [0,%d) "
                "center=%dx%d@(%d,%d) pattern=%d N=%d\n",
                plan.height_in, plan.width_in, kImgPixels,
                plan.crop_h, plan.crop_w, plan.crop_row0, plan.crop_col0,
                plan.pattern_length, plan.N);
        }
        else
        {
            std::printf("cascade_mnist: PadLow pattern=%d N=%d pad_tail=%d\n",
                        plan.pattern_length, plan.N,
                        plan.N - plan.pattern_length);
        }
        const auto noise_grid = MakeTestNoiseGrid();
        PrintTestNoiseReport(noise_grid);
        std::fflush(stdout);

        std::vector<float> train_fields;
        std::vector<int> train_labels;
        std::vector<float> test_fields;
        std::vector<int> test_labels;
        PackSet(train, emb, train_fields, train_labels);
        PackSet(test, emb, test_fields, test_labels);

        Cascade cas(base);
        PathResult run;
        TrainPath("cascade_mnist", cas, train_fields, train_labels, run);

        const bool sweep = kTestNoiseSweep;
        bool check_floor = true;
        std::vector<double> accs;
        std::vector<double> secs;
        if (sweep)
        {
            const double clean_acc = RunNoiseSweep(
                cas, run.name, run.train_acc, run.secs_collect_train,
                test_fields, test_labels, noise_grid, accs, secs);
            if (clean_acc >= 0.0)
                run.test_acc = clean_acc;
            else
                check_floor = false;
        }
        else
        {
            auto t0 = std::chrono::steady_clock::now();
            run.test_acc = ScoreNoisyTest(
                cas, test_fields, test_labels, 0.0f);
            auto t1 = std::chrono::steady_clock::now();
            run.secs_test = std::chrono::duration<double>(t1 - t0).count();
            PrintPathSummary(run);
        }

        if (check_floor && run.test_acc < kMinTestAcc)
        {
            std::fprintf(stderr,
                         "cascade_mnist: test accuracy too low "
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
        std::fprintf(stderr, "cascade_mnist: %s\n", e.what());
        std::fprintf(stderr,
                     "Place uncompressed MNIST IDX files in "
                     "C:\\HypercubeCascade\\data:\n"
                     "  train-images-idx3-ubyte  train-labels-idx1-ubyte\n"
                     "  t10k-images-idx3-ubyte   t10k-labels-idx1-ubyte\n");
    }
    return exit_code;
}
