#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>
#include <span>
#include <vector>

/// @brief Construction-time parameters for @ref Reservoir.
///
/// All fields are fixed at @ref Reservoir::Create. Dynamics (state, history,
/// staged drive) are not part of this struct. @ref GetConfig returns an
/// equivalent snapshot; @c spectral_radius is the **target** used for the
/// recurrent rescale, not the realized estimate.
///
/// No external-feedback port. No multi-channel block inject — drive is a
/// length-N field via @ref InjectInputField.
struct ReservoirConfig
{
    /// Hypercube dimension; neuron count N = 2^dim. Valid range **[5, 16]**.
    size_t dim = 10;

    /// Master RNG seed for weight / bias draws (named substreams in Reservoir.cpp).
    uint64_t seed = 7934791766227647176;

    /// Target spectral radius for the **recurrent** weight block only (> 0).
    float spectral_radius = 0.999f;

    /// Leaky-integrator mix: 1 = full replacement each step; in (0, 1] blends.
    float leak_rate = 1.0f;

    /// Input drive strength. Input weights U(-1,1) then × input_scaling / √dim.
    float input_scaling = 0.02f;

    /// Delay-line length M. Valid range **[1, 64]**.
    size_t history_depth = 16;

    /// If true, print one construction banner.
    bool verbose = false;

    /// Per-neuron bias: U(-1,1) × bias_scaling after tanh. **0** disables.
    float bias_scaling = 0.003f;
};

/// @brief Fixed recurrent core: N = 2^dim neurons on a Boolean hypercube.
///
/// HypercubeESN-derived recurrent core: cube topology, delay line, W_in
/// gather, spectral-radius rescale. **No** external-feedback path. Drive
/// via @ref InjectInputField (length N) only.
///
/// Per-step contract:
/// ```
///   InjectInputField(x, N);
///   Step();
///   // read Outputs() / SliceAt(age)
/// ```
/// Staged input is consumed and zeroed by every @ref Step.
///
/// Non-copyable; obtain instances only via @ref Create.
class Reservoir
{
public:
    static constexpr uint32_t NearestMask(size_t i) { return 1u << i; }

    static std::unique_ptr<Reservoir> Create(const ReservoirConfig& cfg)
    {
        return std::unique_ptr<Reservoir>(new Reservoir(cfg));
    }

    Reservoir(const Reservoir&) = delete;
    Reservoir& operator=(const Reservoir&) = delete;

    /// Advance one step: update vertices, age delay line, clear staged input.
    void Step();

    /// Stage a full length-N input field for the next @ref Step.
    /// @throws std::invalid_argument if @p count != N or @p field is null.
    void InjectInputField(const float* field, size_t count);

    /// Zero dynamical state and history; re-home the slice ring (canonical ages).
    /// Weights and bias unchanged. Prefer @ref LoadInitialCondition for episodes.
    void Clear();

    /// Load a full delay-line IC: @p ic length must be N * history_depth, logical
    /// age order (age 0 = newest). Resets ring to canonical layout and copies
    /// age-0 into the live write state. Intended for frozen s0 each episode.
    /// @throws std::invalid_argument if @p count != N * M or @p ic is null.
    void LoadInitialCondition(const float* ic, size_t count);

    [[nodiscard]] const float* Outputs() const { return slice_ptrs_[0]; }

    /// Logical age slice; SliceAt(0) ≡ Outputs().
    /// @throws std::out_of_range if age >= history_depth.
    [[nodiscard]] const float* SliceAt(size_t age) const;

    [[nodiscard]] float GetRealizedSpectralRadius() const { return realized_spectral_radius_; }

    [[nodiscard]] ReservoirConfig GetConfig() const;

    [[nodiscard]] size_t Dim() const { return dim_; }

    [[nodiscard]] size_t Size() const { return n_; }

    [[nodiscard]] size_t HistoryDepth() const { return history_depth_; }

    struct Snapshot
    {
        std::vector<float> state;   ///< N floats
        std::vector<float> history; ///< N * history_depth, newest first
    };

    [[nodiscard]] Snapshot TakeSnapshot() const;

    void RestoreSnapshot(const Snapshot& snap);

private:
    explicit Reservoir(const ReservoirConfig& cfg);

    struct AlignedFree
    {
        void operator()(float* p) const noexcept
        {
            ::operator delete[](p, std::align_val_t{64});
        }
    };

    static float* AllocAligned(size_t count)
    {
        return static_cast<float*>(
            ::operator new[](count * sizeof(float), std::align_val_t{64}));
    }

    uint64_t rng_seed_ = 0;

    size_t dim_ = 0;
    size_t n_ = 0;
    size_t num_input_weights_ = 0;

    std::unique_ptr<float[], AlignedFree> vtx_input_;
    std::unique_ptr<float[], AlignedFree> vtx_state_;
    std::unique_ptr<float[], AlignedFree> vtx_output_history_;
    std::unique_ptr<float[], AlignedFree> vtx_weight_;
    std::unique_ptr<float*[]> slice_ptrs_;
    std::unique_ptr<float[], AlignedFree> vtx_bias_;

    float spectral_radius_ = 0.99f;
    float leak_rate_ = 1.0f;
    float input_scaling_ = 0.5f;
    float realized_spectral_radius_ = 0.0f;
    bool verbose_ = false;
    size_t history_depth_ = 1;
    size_t num_weights_ = 0;

    float bias_scaling_ = 0.0f;

    void Initialize();
    void UpdateState(size_t v, float old_output_v);
    [[nodiscard]] float EstimateSpectralRadius(std::span<float> x, std::span<float> y) const;

    /// Recurrent block starts after the input block (no ext-fb).
    [[nodiscard]] size_t RecurrentWeightBase() const { return num_input_weights_; }

    void HomeSlicePointers();
};
