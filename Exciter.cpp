#include "Exciter.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <stdexcept>

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

Exciter::Exciter(const ExciterConfig& cfg)
    : rng_seed_(cfg.seed),
      dim_(cfg.dim),
      subcube_dim_(cfg.subcube_dim),
      input_scaling_(cfg.input_scaling),
      weight_scaling_(cfg.weight_scaling)
{
    if (dim_ < 4 || dim_ > 12)
        throw std::invalid_argument("Exciter: dim must be in 4 <= dim <= 12");
    if (subcube_dim_ < 1 || subcube_dim_ > dim_)
    {
        throw std::invalid_argument(
            "Exciter: subcube_dim must be in 1 <= subcube_dim <= dim "
            "(walk at least 2 vertices)");
    }

    n_ = 1ULL << dim_;
    m_ = 1ULL << subcube_dim_;

    state_.assign(n_, 0.0f);
    output_.assign(n_, 0.0f);
    weight_.assign(n_ * dim_, 0.0f);

    Initialize();
}

// ---------------------------------------------------------------------------
// Weight draw
// ---------------------------------------------------------------------------

void Exciter::Initialize()
{
    std::mt19937_64 rng(rng_seed_);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    for (size_t i = 0; i < n_ * dim_; ++i)
        weight_[i] = static_cast<float>(dist(rng)) * weight_scaling_;
}

// ---------------------------------------------------------------------------
// Dynamics
// ---------------------------------------------------------------------------

const float* Exciter::ExciteCube(float* input_field)
{
    if (input_field == nullptr)
        throw std::invalid_argument("ExciteCube: input_field is null");

    // Scale once; every rotation reloads this same IC into state_.
    for (size_t i = 0; i < n_; ++i)
        input_field[i] *= input_scaling_;

    for (size_t r = 0; r < n_; ++r)
    {
        std::memcpy(state_.data(), input_field, n_ * sizeof(float));
        output_[r] = ExciteRotation(r);
    }

    return output_.data();
}

void Exciter::UpdateSite(const size_t vv)
{
    float sum = 0.0f;
    const float* w = weight_.data() + vv * dim_;
    for (size_t j = 0; j < dim_; ++j)
        sum += state_[vv ^ NearestMask(j)] * w[j];
    state_[vv] = std::tanh(sum);
}

float Exciter::ExciteRotation(const size_t r)
{
    const auto m = static_cast<int64_t>(m_);

    // Forward: v = 0 .. M-1 (ends at the face antipodal reflector of r).
    for (int64_t v = 0; v < m; ++v)
        UpdateSite(static_cast<size_t>(v) ^ r);

    // Reverse: skip v = M-1 (just written). Full star — off-face taps
    // still read the scaled IC, which this walk never updates.
    for (int64_t v = m - 2; v >= 0; --v)
        UpdateSite(static_cast<size_t>(v) ^ r);

    return state_[r];
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

ExciterConfig Exciter::GetConfig() const
{
    ExciterConfig cfg;
    cfg.dim = dim_;
    cfg.seed = rng_seed_;
    cfg.input_scaling = input_scaling_;
    cfg.weight_scaling = weight_scaling_;
    cfg.subcube_dim = subcube_dim_;
    return cfg;
}
