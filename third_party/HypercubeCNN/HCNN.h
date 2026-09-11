// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 David Liptak

#pragma once

#include "HCNNTypes.h"
#include "HCNNInput.h"

#include <memory>
#include <vector>

namespace hcnn {

class HCNNNetwork;  // private PIMPL (not installed; not for apps)

/// @brief Bundle of common train-call knobs (learning rate, regularization, shuffle).
///
/// Prefer these overloads over the long positional argument lists:
///
///     TrainParams p;
///     p.learning_rate = 1e-3f;
///     p.weight_decay  = 1e-3f;
///     p.shuffle_seed  = static_cast<unsigned>(epoch + 1);
///     net.TrainEpoch(x, len, y, n, batch, p);
///
/// Defaults match the in-tree examples (Adam-friendly: lr 1e-3, no momentum).
struct TrainParams {
    float learning_rate = 1e-3f;
    float momentum      = 0.0f;   ///< SGD momentum; ignored by Adam
    float weight_decay  = 0.0f;   ///< L2 / AdamW-style decoupled decay on kernels
    /// Classification only: optional per-class loss scale (length GetNumOutputs()).
    const float* class_weights = nullptr;
    /// Epoch only: 0 = sequential zero-copy; nonzero = deterministic shuffle seed.
    unsigned shuffle_seed = 0;
};

/// @class HCNN
/// @brief Public HypercubeCNN front door — the integration surface.
///
/// Dependency-free C++23 hypercube CNN core for research and systems
/// integration.  One class wraps the pipeline: input embedding → conv/pool
/// stack → readout, plus single-sample and batch inference/training for
/// classification and regression.  The public surface is small and
/// contract-driven; optional helpers and examples are not required to use it.
///
/// **Include this header** for core-only integrations, or the umbrella
/// `HypercubeCNN.h` for the full public stack.  `HCNNNetwork` and layer
/// headers are **private** (not installed); application code should never
/// include them.
///
/// Build incrementally with AddConv()/AddPool(), then RandomizeWeights():
///
///     hcnn::HCNN net(10, /*num_outputs=*/10);   // DIM=10, N=1024, 10 classes
///     net.AddConv(32);
///     net.AddPool(hcnn::PoolType::MAX);
///     net.AddConv(64);
///     net.RandomizeWeights();
///     // Default optimizer is Adam — no SetOptimizer required for demos.
///
///     std::vector<float> logits(net.GetNumOutputs());
///     net.Predict(raw, raw_len, logits.data());  // embed + forward
///     int pred = net.PredictClass(raw, raw_len); // classification only
///
/// Regression: pass `TaskType::Regression` and use `*Regression` train APIs
/// (or `TrainParams` overloads of those methods).
///
/// **Contiguous data model.**  Batch/epoch methods take row-major
/// `const float*` bases + uniform `input_length`.
/// Prefer **`HCNNInputView` / `HCNNInputBatch`** (full capacity per sample)
/// after spatial embed so short `input_length` cannot wipe non-zero pad.
///
/// **Non-copyable, movable.**  The live thread pool lives on the heap
/// (`unique_ptr<HCNNNetwork>`); move transfers ownership without relocating
/// workers.  Moved-from objects are empty and must not be used.
class HCNN {
public:
    /// @param num_threads 0 = auto, 1 = single-threaded (no pool workers), N = N workers.
    /// @param start_dim Hypercube dimension in [3, 30].
    /// @param task_type Classification → CE; Regression → MSE (fixed pairing).
    explicit HCNN(int start_dim, int num_outputs = 10,
                  int input_channels = 1,
                  TaskType task_type = TaskType::Classification,
                  size_t num_threads = 0);
    ~HCNN();

    HCNN(const HCNN&) = delete;
    HCNN& operator=(const HCNN&) = delete;
    HCNN(HCNN&&) noexcept;
    HCNN& operator=(HCNN&&) noexcept;

    // -----------------------------------------------------------------
    //  Architecture (incremental builder)
    // -----------------------------------------------------------------

    /// Append a conv layer.  Invalidates weights — call RandomizeWeights again
    /// before train/infer if the net was already randomized.
    void AddConv(int c_out, Activation activation = Activation::RELU,
                 bool use_bias = true, bool use_batchnorm = false);

    /// Antipodal pool; reduces DIM by 1.  Same weight-invalidation rule as AddConv.
    void AddPool(PoolType type = PoolType::MAX);

    /// Initialize all weights and size the FLATTEN head to the current stack.
    /// Required after construct (with ≥1 conv) and after any AddConv/AddPool.
    /// scale > 0: uniform [-scale, +scale]; scale <= 0: per-layer Xavier/He.
    /// @p seed is a full 64-bit master seed (see HCNNNetwork::randomize_all_weights).
    void RandomizeWeights(float scale = 0.0f, uint64_t seed = 42);

    // -----------------------------------------------------------------
    //  Mode / optimizer
    // -----------------------------------------------------------------

    void SetTraining(bool training);

    /// Configure optimizer for all layers.  Default at construction is Adam.
    /// Resets Adam timestep and re-inits second-moment buffers.  Survives
    /// RandomizeWeights.
    void SetOptimizer(OptimizerType type, float beta1 = 0.9f,
                      float beta2 = 0.999f, float eps = 1e-8f);

    /// Session train knobs used by train overloads that omit `TrainParams`.
    /// Survives RandomizeWeights / SetOptimizer.  Default: lr 1e-3, rest zero.
    void SetTrainDefaults(const TrainParams& params);
    [[nodiscard]] const TrainParams& GetTrainDefaults() const;

    /// Eagerly allocate work buffers (prefer after RandomizeWeights). Idempotent.
    void PrepareBuffers();

    // -----------------------------------------------------------------
    //  Inference
    // -----------------------------------------------------------------

    /// Map raw floats onto capacity = input_channels * GetStartN() (zero-pad tail).
    /// `embedded_out` must hold that many floats (caller-owned).
    void Embed(const float* raw_input, int input_length,
               float* embedded_out) const;

    /// Forward from already-embedded activations.  No allocation.
    /// Outputs: raw logits (classification) or predictions (regression).
    void Forward(const float* embedded, float* logits) const;

    /// Embed + Forward in one call.  Reuses internal scratch (not concurrent
    /// with other calls on this instance).
    void Predict(const float* raw_input, int input_length, float* outputs) const;

    /// Full-capacity single sample (`in.count() == 1`).  See HCNNInput.h.
    void Predict(HCNNInputView in, float* outputs) const;

    /// Classification only: Embed + Forward + argmax.  Throws logic_error
    /// on Regression nets.
    [[nodiscard]] int PredictClass(const float* raw_input, int input_length) const;

    /// Full-capacity single sample (`in.count() == 1`).
    [[nodiscard]] int PredictClass(HCNNInputView in) const;

    /// Batch inference (parallel).  `flat_inputs`: batch_size * input_length;
    /// `logits_out`: batch_size * GetNumOutputs().
    void ForwardBatch(const float* flat_inputs, int input_length,
                      int batch_size, float* logits_out);

    /// Full-capacity batch; `logits_out` holds count * GetNumOutputs().
    void ForwardBatch(HCNNInputView in, float* logits_out);

    // -----------------------------------------------------------------
    //  Training — one vocabulary; overload by target type
    // -----------------------------------------------------------------
    // Classification: int / const int* class indices; TaskType::Classification
    // Regression:     const float* targets (num_outputs per sample);
    //                 TaskType::Regression
    // Wrong task → std::logic_error.
    //
    // Prefer TrainParams or SetTrainDefaults + no-param overloads.
    // Prefer HCNNInputView after spatial pack (full capacity).

    // --- Positional ---
    void TrainStep(const float* raw_input, int input_length, int target_class,
                   float learning_rate, float momentum = 0.0f,
                   float weight_decay = 0.0f,
                   const float* class_weights = nullptr);
    void TrainStep(const float* raw_input, int input_length,
                   const float* target, float learning_rate,
                   float momentum = 0.0f, float weight_decay = 0.0f);

    void TrainBatch(const float* flat_inputs, int input_length,
                    const int* targets, int batch_size,
                    float learning_rate, float momentum = 0.0f,
                    float weight_decay = 0.0f,
                    const float* class_weights = nullptr);
    void TrainBatch(const float* flat_inputs, int input_length,
                    const float* flat_targets, int batch_size,
                    float learning_rate, float momentum = 0.0f,
                    float weight_decay = 0.0f);

    void TrainEpoch(const float* flat_inputs, int input_length,
                    const int* targets, int sample_count, int batch_size,
                    float learning_rate, float momentum = 0.0f,
                    float weight_decay = 0.0f,
                    const float* class_weights = nullptr,
                    unsigned shuffle_seed = 0);
    void TrainEpoch(const float* flat_inputs, int input_length,
                    const float* flat_targets, int sample_count, int batch_size,
                    float learning_rate, float momentum = 0.0f,
                    float weight_decay = 0.0f,
                    unsigned shuffle_seed = 0);

    // --- TrainParams ---
    void TrainStep(const float* raw_input, int input_length, int target_class,
                   const TrainParams& params);
    void TrainStep(const float* raw_input, int input_length,
                   const float* target, const TrainParams& params);
    void TrainBatch(const float* flat_inputs, int input_length,
                    const int* targets, int batch_size,
                    const TrainParams& params);
    void TrainBatch(const float* flat_inputs, int input_length,
                    const float* flat_targets, int batch_size,
                    const TrainParams& params);
    void TrainEpoch(const float* flat_inputs, int input_length,
                    const int* targets, int sample_count, int batch_size,
                    const TrainParams& params);
    void TrainEpoch(const float* flat_inputs, int input_length,
                    const float* flat_targets, int sample_count, int batch_size,
                    const TrainParams& params);

    // --- Session defaults (GetTrainDefaults) ---
    void TrainStep(const float* raw_input, int input_length, int target_class);
    void TrainStep(const float* raw_input, int input_length,
                   const float* target);
    void TrainBatch(const float* flat_inputs, int input_length,
                    const int* targets, int batch_size);
    void TrainBatch(const float* flat_inputs, int input_length,
                    const float* flat_targets, int batch_size);
    void TrainEpoch(const float* flat_inputs, int input_length,
                    const int* targets, int sample_count, int batch_size);
    void TrainEpoch(const float* flat_inputs, int input_length,
                    const float* flat_targets, int sample_count, int batch_size);

    // --- Full-capacity HCNNInputView ---
    void TrainStep(HCNNInputView in, int target_class, const TrainParams& params);
    void TrainStep(HCNNInputView in, const float* target,
                   const TrainParams& params);
    void TrainBatch(HCNNInputView in, const int* targets, int batch_size,
                    const TrainParams& params);
    void TrainBatch(HCNNInputView in, const float* flat_targets, int batch_size,
                    const TrainParams& params);
    void TrainEpoch(HCNNInputView in, const int* targets, int batch_size,
                    const TrainParams& params);
    void TrainEpoch(HCNNInputView in, const float* flat_targets, int batch_size,
                    const TrainParams& params);

    void TrainStep(HCNNInputView in, int target_class);
    void TrainStep(HCNNInputView in, const float* target);
    void TrainBatch(HCNNInputView in, const int* targets, int batch_size);
    void TrainBatch(HCNNInputView in, const float* flat_targets, int batch_size);
    void TrainEpoch(HCNNInputView in, const int* targets, int batch_size);
    void TrainEpoch(HCNNInputView in, const float* flat_targets, int batch_size);

    // -----------------------------------------------------------------
    //  Sizing accessors
    // -----------------------------------------------------------------
    int GetStartDim() const;
    int GetStartN() const;
    int GetCurrentDim() const;
    int GetInputChannels() const;
    int GetNumOutputs() const;
    size_t GetNumConv() const;
    size_t GetNumPool() const;
    TaskType GetTaskType() const;
    OptimizerType GetOptimizerType() const;
    bool WeightsInitialized() const;

    // -----------------------------------------------------------------
    //  Weight serialization
    // -----------------------------------------------------------------

    /// Total floats in the GetWeights blob (requires WeightsInitialized).
    [[nodiscard]] size_t GetWeightCount() const;

    /// Flatten parameters + BN running stats (when present) into caller buffer.
    /// @param out  Must hold at least GetWeightCount() floats.
    /// @param n    Size of `out` in floats; must equal GetWeightCount().
    /// Layout: per conv (kernel, bias?, γ/β/running_mean/running_var?) then
    /// readout weights + bias.  Does **not** include optimizer moments.
    void GetWeights(float* out, size_t n) const;

    /// Allocating convenience wrapper around GetWeights(float*, size_t).
    [[nodiscard]] std::vector<float> GetWeights() const;

    /// Restore from GetWeights layout (pointer form; no extra allocation).
    /// @param reset_optimizer_moments true → zero moments + Adam step (train resume).
    ///        false (default) → weights/BN only (eval/export restore).
    void SetWeights(const float* data, size_t n,
                    bool reset_optimizer_moments = false);

    /// Vector overload of SetWeights(const float*, size_t, bool).
    void SetWeights(const std::vector<float>& blob,
                    bool reset_optimizer_moments = false);

private:
    std::unique_ptr<HCNNNetwork> net_;
    TrainParams train_defaults_;

    std::vector<int> shuffle_idx_;
    std::vector<float> shuffle_inputs_;
    std::vector<int>   shuffle_targets_;
    std::vector<float> shuffle_targets_f_;

    /// Scratch for Predict / PredictClass (capacity + logits).
    mutable std::vector<float> predict_embed_;
    mutable std::vector<float> predict_logits_;

    void require_weights_initialized_(const char* api) const;
    void ensure_predict_buffers_() const;
    /// capacity must equal GetInputChannels() * GetStartN().
    void require_input_view_(HCNNInputView in, const char* api) const;

    template <typename GatherTargets, typename TrainChunk>
    void train_epoch_impl_(const float* flat_inputs, int input_length,
                           int sample_count, int batch_size,
                           unsigned shuffle_seed,
                           GatherTargets&& gather_targets,
                           TrainChunk&& train_chunk);
};

} // namespace hcnn
