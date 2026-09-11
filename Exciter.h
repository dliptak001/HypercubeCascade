#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

/// @brief Construction-time parameters for @ref Exciter.
///
/// Frozen map. Neighbor weights are drawn once and never trained.
struct ExciterConfig
{
    /// Hypercube dimension; field length N = 2^dim. Valid range **[4, 12]**:
    /// 4 is the smallest cube with a useful neighbor star; 12 is the cost cap
    /// (N = 4096).
    size_t dim = 8;

    /// Master RNG seed for neighbor weight draws.
    uint64_t seed = 7934791766227647176;

    /// Scalar gain applied once in place to the input field in @ref ExciteCube.
    /// A second call on the same buffer applies this gain again.
    float input_scaling = 0.02f;

    /// Scale on neighbor weight draws: U(-1, 1) × weight_scaling.
    float weight_scaling = 0.02f;

    /// Dimension of the face each start covers in an etalon transit.
    /// M = 2^subcube_dim vertices per start. Valid **[1, dim]**. `dim`
    /// is the whole cube; `dim-1` is a half-cube. Pins the high
    /// `dim - subcube_dim` bits; the transit along that face goes
    /// through r. Full star: off-face neighbors stay at the scaled
    /// input. Default 6 with default dim 8 is M = 64.
    size_t subcube_dim = 6;
};

/// @brief Frozen hypercube map: length-N field → length-N field.
///
/// Neighbor weights are drawn once from U(-1, 1) × weight_scaling and
/// never updated. This is not a reservoir: there is no leak, no orbit,
/// and no delay line.
///
/// @ref ExciteCube is one etalon transit. Each start vertex r and its
/// face antipode are a pair of reflectors. The transit scales the input
/// once, then for every r reloads that scaled field and visits
/// v = 0 … M-1 then M-2 … 0 of the physical vertex `v xor r`. Each site
/// write is tanh of the full-star neighbor sum. The value written on the
/// second visit to r is the output sample.
class Exciter
{
public:
    /// @brief Heap-allocate an Exciter.
    /// @throws std::invalid_argument if @c dim is not in [4, 12] or
    ///         @c subcube_dim is not in [1, dim].
    static std::unique_ptr<Exciter> Create(const ExciterConfig& cfg)
    {
        return std::unique_ptr<Exciter>(new Exciter(cfg));
    }

    Exciter(const Exciter&) = delete;
    Exciter& operator=(const Exciter&) = delete;

    /// One etalon transit: scale @p input_field in place by
    /// input_scaling, then for each start r reload that field, go to
    /// the face antipode and back, and write output[r]. Length must be
    /// N(); must not alias internals.
    /// @return length-N output; valid until the next ExciteCube or destroy.
    /// @throws std::invalid_argument if @p input_field is null.
    [[nodiscard]] const float* ExciteCube(float* input_field);

    [[nodiscard]] ExciterConfig GetConfig() const;

    [[nodiscard]] size_t Dim() const { return dim_; }

    [[nodiscard]] size_t N() const { return n_; }
    [[nodiscard]] size_t Size() const { return n_; } ///< Same as @ref N.

    [[nodiscard]] size_t SubcubeDim() const { return subcube_dim_; }

    /// Vertices on one start's out-and-back: M = 2^subcube_dim.
    [[nodiscard]] size_t WalkSize() const { return m_; }

private:
    static constexpr uint32_t NearestMask(size_t i) { return 1u << i; }

    explicit Exciter(const ExciterConfig& cfg);

    uint64_t rng_seed_ = 0;

    size_t dim_ = 0;
    size_t n_ = 0;
    size_t subcube_dim_ = 0;
    size_t m_ = 0; ///< 2^subcube_dim

    std::vector<float> state_;
    std::vector<float> output_;
    std::vector<float> weight_; ///< neighbor: N · dim

    float input_scaling_ = 0.02f;
    float weight_scaling_ = 0.02f;

    void Initialize();
    void UpdateSite(size_t vv);
    float ExciteRotation(size_t r);
};
