#include "Reservoir.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <new>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

Reservoir::Reservoir(const ReservoirConfig& cfg)
    : rng_seed_(cfg.seed),
      dim_(cfg.dim),
      spectral_radius_(cfg.spectral_radius),
      leak_rate_(cfg.leak_rate),
      input_scaling_(cfg.input_scaling),
      verbose_(cfg.verbose),
      history_depth_(cfg.history_depth),
      bias_scaling_(cfg.bias_scaling)
{
    if (dim_ < 5 || dim_ > 16)
        throw std::invalid_argument("dim must be in 5 <= dim <= 16");

    n_ = 1ULL << dim_;
    num_input_weights_ = n_ * dim_;

    if (spectral_radius_ <= 0.0f)
        throw std::invalid_argument("spectral_radius must be positive");
    if (leak_rate_ <= 0.0f || leak_rate_ > 1.0f)
        throw std::invalid_argument("leak_rate must be in (0.0, 1.0]");
    if (history_depth_ < 1 || history_depth_ > 64)
        throw std::invalid_argument("history_depth must be in [1, 64]");

    // Weight layout: [ input: N·DIM | recurrent: N·M·DIM ]
    num_weights_ = n_ * dim_ * (history_depth_ + 1u);

    vtx_input_.reset(AllocAligned(n_));
    vtx_state_.reset(AllocAligned(n_));
    vtx_output_history_.reset(AllocAligned(n_ * history_depth_));
    vtx_weight_.reset(AllocAligned(num_weights_));
    slice_ptrs_.reset(new float*[history_depth_]());
    vtx_bias_.reset(AllocAligned(n_));

    Initialize();
}

// ---------------------------------------------------------------------------
// Seeding
// ---------------------------------------------------------------------------

static inline uint64_t mix64(uint64_t x)
{
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

// Named substreams (values are part of the weight-draw ABI vs hESN roles).
enum class SeedRole : uint64_t {
    Recurrent = 1,
    Input = 2,
    // 3 reserved (was ExternalFeedback in hESN) — never reuse for a new role
    // that should not collide with historical draws if code is compared.
    Bias = 4,
    SrProbe = 5
};

// ---------------------------------------------------------------------------
// Weight draw + spectral-radius rescale
// ---------------------------------------------------------------------------

void Reservoir::Initialize()
{
    auto seed_for = [this](SeedRole r) {
        return mix64(rng_seed_ ^ (0x100000001B3ULL * static_cast<uint64_t>(r)));
    };
    std::mt19937_64 rng(seed_for(SeedRole::Recurrent));
    std::mt19937_64 in_rng(seed_for(SeedRole::Input));
    std::mt19937_64 bias_rng(seed_for(SeedRole::Bias));
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    Clear();

    for (size_t i = 0; i < n_; ++i)
        vtx_bias_[i] = static_cast<float>(dist(bias_rng)) * bias_scaling_;

    float* pW = vtx_weight_.get();

    float* const input_base = pW;
    for (size_t i = 0; i < num_input_weights_; ++i)
        (*pW++) = static_cast<float>(dist(in_rng));
    const float in_scaling = input_scaling_ / std::sqrt(static_cast<float>(dim_));
    for (size_t i = 0; i < num_input_weights_; ++i)
        input_base[i] *= in_scaling;

    const size_t rec_base = RecurrentWeightBase();
    const float w_scaling =
        1.0f / std::sqrt(static_cast<float>(dim_ * history_depth_));
    for (size_t i = rec_base; i < num_weights_; ++i)
        vtx_weight_[i] = static_cast<float>(dist(rng)) * w_scaling;

    const float target = spectral_radius_;
    const size_t MN = history_depth_ * n_;
    std::vector<float> sr_x(MN, 0.0f), sr_y(MN, 0.0f);
    {
        std::mt19937_64 sr_rng(seed_for(SeedRole::SrProbe));
        std::uniform_real_distribution<double> sr_dist(-1.0, 1.0);
        float norm = 0.0f;
        for (size_t v = 0; v < n_; ++v)
        {
            sr_x[v] = static_cast<float>(sr_dist(sr_rng));
            norm += sr_x[v] * sr_x[v];
        }
        norm = std::sqrt(norm);
        for (size_t v = 0; v < n_; ++v)
            sr_x[v] /= norm;
    }

    float applied_scale = 1.0f;
    auto eval_sr = [&](float s) {
        const float rel = s / applied_scale;
        for (size_t i = rec_base; i < num_weights_; ++i)
            vtx_weight_[i] *= rel;
        applied_scale = s;
        return EstimateSpectralRadius(sr_x, sr_y);
    };

    const float pre_sr = EstimateSpectralRadius(sr_x, sr_y);
    float post_sr = pre_sr;
    int sr_iters = 0;
    if (pre_sr > 1e-6f)
    {
        constexpr float kSrTolRel = 0.001f;
        constexpr int kMaxSrIters = 20;

        float s0 = 1.0f, h0 = pre_sr - target;
        float s1 = target / pre_sr, h1 = eval_sr(s1) - target;
        ++sr_iters;
        post_sr = h1 + target;
        while (sr_iters < kMaxSrIters &&
               std::abs(post_sr - target) > target * kSrTolRel)
        {
            const float denom = h1 - h0;
            float s2 = (std::abs(denom) < 1e-12f)
                           ? s1 * (target / std::max(post_sr, 1e-6f))
                           : s1 - h1 * (s1 - s0) / denom;
            s2 = std::clamp(s2, 0.25f * s1, 4.0f * s1);
            post_sr = eval_sr(s2);
            ++sr_iters;
            s0 = s1;
            h0 = h1;
            s1 = s2;
            h1 = post_sr - target;
        }
    }
    realized_spectral_radius_ = post_sr;
    if (verbose_)
    {
        std::printf("[Reservoir DIM=%zu M=%zu seed=%llu leak=%.3g in_scale=%.3g "
                    "SR target=%.4f post=%.4f (secant iters=%d)]\n",
                    dim_, history_depth_,
                    static_cast<unsigned long long>(rng_seed_), leak_rate_,
                    input_scaling_, target, post_sr, sr_iters);
    }
}

// ---------------------------------------------------------------------------
// Dynamics
// ---------------------------------------------------------------------------

void Reservoir::Step()
{
    const float* p_vtx_prev = slice_ptrs_[0];
    for (size_t v = 0; v < n_; v++)
        UpdateState(v, p_vtx_prev[v]);

    float* p0 = slice_ptrs_[history_depth_ - 1];
    for (size_t i = history_depth_ - 1; i > 0; --i)
        slice_ptrs_[i] = slice_ptrs_[i - 1];
    slice_ptrs_[0] = p0;

    std::memcpy(slice_ptrs_[0], vtx_state_.get(), n_ * sizeof(float));
    std::memset(vtx_input_.get(), 0, n_ * sizeof(float));
}

void Reservoir::UpdateState(const size_t v, const float old_output_v)
{
    float s = 0.0f;
    const float* iw = vtx_weight_.get() + v * dim_;
    const float* w =
        &vtx_weight_[RecurrentWeightBase()] + v * dim_ * history_depth_;

    for (size_t i = 0; i < dim_; i++)
        s += vtx_input_[v ^ NearestMask(i)] * iw[i];

    for (size_t i = 0; i < history_depth_; i++)
    {
        const float* pSlice = slice_ptrs_[i];
        for (size_t j = 0; j < dim_; j++)
            s += pSlice[v ^ NearestMask(j)] * (*w++);
    }

    const float activation = std::tanh(s) + vtx_bias_[v];
    vtx_state_[v] = (1.0f - leak_rate_) * old_output_v + leak_rate_ * activation;
}

// ---------------------------------------------------------------------------
// Drive injection / IC
// ---------------------------------------------------------------------------

void Reservoir::InjectInputField(const float* field, const size_t count)
{
    if (field == nullptr)
        throw std::invalid_argument("InjectInputField: field is null");
    if (count != n_)
        throw std::invalid_argument(
            "InjectInputField: count must equal N = 2^dim");
    std::memcpy(vtx_input_.get(), field, n_ * sizeof(float));
}

void Reservoir::HomeSlicePointers()
{
    for (size_t i = 0; i < history_depth_; i++)
        slice_ptrs_[i] = &vtx_output_history_[i * n_];
}

void Reservoir::LoadInitialCondition(const float* ic, const size_t count)
{
    if (ic == nullptr)
        throw std::invalid_argument("LoadInitialCondition: ic is null");
    const size_t need = n_ * history_depth_;
    if (count != need)
        throw std::invalid_argument(
            "LoadInitialCondition: count must equal N * history_depth");

    // Canonical ring home, then bulk load logical ages 0..M-1 into physical slots.
    HomeSlicePointers();
    std::memcpy(vtx_output_history_.get(), ic, need * sizeof(float));
    std::memcpy(vtx_state_.get(), ic, n_ * sizeof(float)); // age-0
    std::memset(vtx_input_.get(), 0, n_ * sizeof(float));
}

// ---------------------------------------------------------------------------
// Snapshot / config / clear
// ---------------------------------------------------------------------------

Reservoir::Snapshot Reservoir::TakeSnapshot() const
{
    Snapshot s;
    s.state.assign(vtx_state_.get(), vtx_state_.get() + n_);
    s.history.resize(n_ * history_depth_);
    for (size_t i = 0; i < history_depth_; ++i)
        std::memcpy(s.history.data() + i * n_, slice_ptrs_[i], n_ * sizeof(float));
    return s;
}

void Reservoir::RestoreSnapshot(const Snapshot& snap)
{
    if (snap.state.size() != n_ || snap.history.size() != n_ * history_depth_)
        throw std::invalid_argument(
            "RestoreSnapshot: snapshot sizes do not match this reservoir "
            "(expected state=N, history=N*history_depth)");

    std::memcpy(vtx_state_.get(), snap.state.data(), n_ * sizeof(float));
    std::memcpy(vtx_output_history_.get(), snap.history.data(),
                n_ * history_depth_ * sizeof(float));
    HomeSlicePointers();
    std::memset(vtx_input_.get(), 0, n_ * sizeof(float));
}

ReservoirConfig Reservoir::GetConfig() const
{
    ReservoirConfig cfg;
    cfg.dim = dim_;
    cfg.seed = rng_seed_;
    cfg.spectral_radius = spectral_radius_;
    cfg.leak_rate = leak_rate_;
    cfg.input_scaling = input_scaling_;
    cfg.history_depth = history_depth_;
    cfg.verbose = verbose_;
    cfg.bias_scaling = bias_scaling_;
    return cfg;
}

const float* Reservoir::SliceAt(const size_t age) const
{
    if (age >= history_depth_)
        throw std::out_of_range(
            "Reservoir::SliceAt: age (" + std::to_string(age) +
            ") >= history_depth (" + std::to_string(history_depth_) + ")");
    return slice_ptrs_[age];
}

void Reservoir::Clear()
{
    std::memset(vtx_state_.get(), 0, n_ * sizeof(float));
    std::memset(vtx_input_.get(), 0, n_ * sizeof(float));
    std::memset(vtx_output_history_.get(), 0, n_ * history_depth_ * sizeof(float));
    HomeSlicePointers();
}

// ---------------------------------------------------------------------------
// Spectral radius (companion operator on MN-dimensional delay state)
// ---------------------------------------------------------------------------

float Reservoir::EstimateSpectralRadius(std::span<float> x, std::span<float> y) const
{
    const size_t MN = history_depth_ * n_;
    assert(x.size() >= MN && y.size() >= MN);

    constexpr int kMaxIters = 1500;
    constexpr int kBurnIn = 32;
    constexpr int kCheckSpacing = 50;
    constexpr float kTolRel = 1e-4f;

    float rho_ring[kCheckSpacing] = {};
    double sum_log = 0.0;
    int n_acc = 0;
    float rho = 0.0f;

    for (int iter = 0; iter < kMaxIters; ++iter)
    {
        for (size_t v = 0; v < n_; v++)
        {
            float s = 0.0f;
            const float* w =
                &vtx_weight_[RecurrentWeightBase()] + v * dim_ * history_depth_;
            for (size_t j = 0; j < history_depth_; j++)
            {
                const float* x_j = x.data() + j * n_;
                const float* wj = w + j * dim_;
                for (size_t i = 0; i < dim_; i++)
                    s += wj[i] * x_j[v ^ NearestMask(i)];
            }
            y[v] = s;
        }

        for (size_t j = 1; j < history_depth_; j++)
            std::memcpy(y.data() + j * n_, x.data() + (j - 1) * n_,
                        n_ * sizeof(float));

        float norm = 0.0f;
        for (size_t k = 0; k < MN; k++)
            norm += y[k] * y[k];
        norm = std::sqrt(norm);
        if (norm <= 1e-30f)
            return 0.0f;

        const float inv = 1.0f / norm;
        for (size_t k = 0; k < MN; k++)
            x[k] = y[k] * inv;

        if (iter < kBurnIn)
            continue;

        sum_log += std::log(static_cast<double>(norm));
        ++n_acc;
        rho = static_cast<float>(std::exp(sum_log / static_cast<double>(n_acc)));

        const int slot = n_acc % kCheckSpacing;
        if (n_acc > kCheckSpacing &&
            std::abs(rho - rho_ring[slot]) < rho * kTolRel)
            break;
        rho_ring[slot] = rho;
    }

    return rho;
}
