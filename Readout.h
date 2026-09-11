#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace hcnn
{
    class HCNN;
}

/// Which kind of task the readout learns; fixed at construction. Regression
/// predicts continuous values; Classification predicts a discrete class label.
enum class ReadoutTask { Regression, Classification };

/// Activation applied after each Conv layer in the Readout's CNN stack.
/// Mirrors `hcnn::Activation` to keep HCNN.h out of this public header
/// (PIMPL discipline -- mapping lives in Readout.cpp).
enum class ReadoutActivation { TANH, RELU, LEAKY_RELU, NONE };

/// Antipodal pool reduction (only used when @ref ReadoutConfig::use_pooling).
enum class ReadoutPoolType { Max, Avg };

/// Weight-update rule forwarded to HypercubeCNN (default Adam).
enum class ReadoutOptimizer { Adam, Sgd };

/// How @ref Readout::SetState treats optimizer state after loading weights.
/// Eval: restore parameters only (default; safe for inference / export).
/// ResumeTrain: also zero Adam/SGD moments so online training continues cleanly.
enum class ReadoutLoadMode { Eval, ResumeTrain };

/// @brief Architecture and training settings for the @ref Readout CNN.
///
/// Frozen knobs — do not add more training-loop policy here. New work
/// belongs on @ref Exciter / @ref Reservoir. For a different train loop,
/// drive HypercubeCNN directly.
///
/// Set @c dim, @c num_outputs, and @c task first. The rest is a fixed
/// HCNN stack plus a cosine Adam/SGD fit. Trivially copyable.
struct ReadoutConfig
{
    /// Input feature dim: features per sample = 2^dim. Valid range **[3, 30]**
    /// (HypercubeCNN). 0 is unset — construction throws. The host sets
    /// this from the field source (@ref Exciter or @ref Reservoir).
    size_t dim = 0;
    int num_outputs = 1; ///< Classes (classification) or targets (regression).
    ReadoutTask task = ReadoutTask::Regression;
    int num_layers = 1; ///< Conv(+Pool) layers. Default 1. 0 = auto: max(1, min(dim-2, 2)).

    /// Append an antipodal pool after each Conv (true = the historical behavior).
    /// The pool pairs each vertex with its bitwise complement, so it mixes *every* bit —
    /// including any block-index bits of a block-structured input. Set false to keep
    /// that structure intact through the conv stack; the flatten readout then sees
    /// twice as many features.
    bool use_pooling = true;
    ReadoutPoolType pool_type = ReadoutPoolType::Max; ///< Used when use_pooling.
    int conv_channels = 16; ///< Base channels for the first conv.
    /// Channel multiplier after each conv stage (historical default: double).
    int channel_growth = 2;
    /// Per-conv batch-norm (HypercubeCNN). Default off — enables BN γ/β + running
    /// stats in the weight blob; keep off for stable checkpoint sizes unless you
    /// intentionally train with BN.
    bool use_batchnorm = false;
    ReadoutOptimizer optimizer = ReadoutOptimizer::Adam;

    int epochs = 200;
    int batch_size = 32;
    float lr_max = 0.0015f; ///< Cosine annealing peak. Keep <= 0.005 to avoid NaN.
    float lr_min_frac = 0.01f; ///< Floor = lr_max * lr_min_frac.
    int lr_decay_epochs = 0; ///< Cosine decay horizon. 0 = use `epochs`.
    float weight_decay = 0.0f;
    float momentum = 0.9f; ///< SGD momentum (heavy-ball). 0 = plain SGD. Ignored by Adam.
    /// CNN weight-init seed (full 64-bit). Forwarded to HypercubeCNN `weight_seed`.
    uint64_t seed = 42;
    ReadoutActivation activation = ReadoutActivation::TANH; ///< Per-Conv-layer activation.

    /// HCNN internal worker-pool size. Forwarded to `hcnn::HCNN`:
    /// 0 = auto (default), 1 = single-threaded (no HCNN background workers),
    /// N > 1 = N workers. Use 1 when the host already parallelizes across
    /// Exciter / Readout instances (e.g. a multi-seed survey) to avoid nested
    /// oversubscription.
    size_t num_threads = 0;

    /// After each batch @ref Readout::Train epoch, score a metric and at the end
    /// restore the best weights seen (regression: min MSE; classification: max
    /// accuracy). Default true (prefer generalization over last-epoch snapshot).
    /// Set false for historical last-epoch behavior. Extra cost: one full forward
    /// over the score set every epoch.
    bool restore_best_epoch = true;
    /// When @c restore_best_epoch is true: fraction of samples held out for
    /// best-metric selection only. Shuffled once with @ref seed (caller
    /// buffers are not mutated). 0 = score the full training set. Clamped
    /// to [0, 0.5]. If the set has fewer than 2 samples, holdout is skipped.
    float best_epoch_holdout_frac = 0.0f;

    /// Optional. Called after each @ref Readout::Train epoch with the 1-based
    /// epoch index and the training-set error (regression: MSE; classification:
    /// mean softmax cross-entropy). Null by default.
    void (*epoch_tick)(int epoch, double train_error) = nullptr;
};

/// @brief Trainable HypercubeCNN façade: length-N field → task outputs.
///
/// A thin HCNN wrapper so the host can collect → train → predict
/// without including `HCNN.h`. Do not add new schedules, checkpoint
/// schemes, or train-loop knobs here. Change the graph via HypercubeCNN;
/// change the map via @ref Exciter or the dynamics via @ref Reservoir.
///
/// Typical input is an @ref Exciter or @ref Reservoir field (still
/// length N). The same façade accepts any length-N field.
///
/// ## Data path
/// ```
///   field[N] ──▶ Embed ──▶ [ Conv (+ Pool) ] × L ──▶ Flatten ──▶ Linear ──▶ output
/// ```
/// @c dim is valid **[3, 30]**. Default @c num_layers is 1. Zero means
/// auto: max(1, min(dim - 2, 2)). With pooling on, @c num_layers must
/// be `<= dim-2`. Prefer @c dim >= 5 so a pooled stack has room.
///
/// ## Lifecycle
/// Collect fields, @ref Train once, then @ref PredictRaw /
/// @ref PredictClass. @ref TrainStep* exist for hosts that already
/// interleave their own loop. Prefer @ref SaveHcnnModel over the
/// unversioned @ref Weights blob.
///
/// @note PIMPL: `HCNN.h` stays out of this header (included only in
///       Readout.cpp).
class Readout
{
public:
    /// @brief Build the CNN immediately from @p cfg.
    /// @throws std::invalid_argument if @c dim is not in [3, 30],
    ///         @c num_outputs / @c conv_channels / @c channel_growth are
    ///         < 1, @c num_layers is < 0, or a pooled stack is deeper
    ///         than @c dim-2.
    explicit Readout(const ReadoutConfig& cfg);
    ~Readout();
    Readout(Readout&&) noexcept;
    Readout& operator=(Readout&&) noexcept;

    Readout(const Readout&) = delete;
    Readout& operator=(const Readout&) = delete;

    // ----- Batch training -----

    /// @brief Batch-train (regression). @p states is sample-major,
    /// length @p num_samples * N; @p targets is @p num_samples *
    /// num_outputs floats. Continues from the current weights — construct
    /// a new Readout for a fresh fit.
    /// @throws std::logic_error if task is Classification.
    void Train(const float* states, const float* targets, size_t num_samples);

    /// @brief Batch-train (classification). @p states is sample-major,
    /// length @p num_samples * N; @p class_labels is @p num_samples ints
    /// in [0, num_outputs). Continues from the current weights.
    /// @throws std::logic_error if task is Regression.
    void Train(const float* states, const int* class_labels, size_t num_samples);

    // ----- Streaming training (thin HCNN forwards; not the product path) -----
    //
    // One gradient step. Host supplies the learning rate. Do not add
    // schedules or extra knobs around these.

    /// @brief One regression step. @p target is num_outputs floats.
    /// @throws std::logic_error if task is Classification.
    void TrainStep(const float* state, const float* target,
                   float lr, float weight_decay = 0.0f);

    /// @brief One classification step. @p class_label in [0, num_outputs).
    /// @throws std::logic_error if task is Regression.
    void TrainStep(const float* state, int class_label,
                   float lr, float weight_decay = 0.0f);

    /// @brief Regression mini-batch. @p targets is count * num_outputs floats.
    /// @throws std::logic_error if task is Classification.
    void TrainStepBatch(const float* states, const float* targets,
                        size_t count, float lr, float weight_decay = 0.0f);

    /// @brief Classification mini-batch. @p class_labels is count ints.
    /// @throws std::logic_error if task is Regression.
    void TrainStepBatch(const float* states, const int* class_labels,
                        size_t count, float lr, float weight_decay = 0.0f);

    // ----- Prediction -----

    /// @brief Run the network on one @p state and write num_outputs floats to
    /// @p output. Regression: the raw network output. Classification: the raw
    /// class logits (use @ref PredictClass for the argmax label).
    void PredictRaw(const float* state, float* output) const;

    /// @brief Predicted class index — the argmax over the classification logits.
    /// @throws std::logic_error if task is Regression.
    [[nodiscard]] int PredictClass(const float* state) const;

    // ----- Evaluation -----

    /// @brief R² (coefficient of determination) over @p num_samples (state, target)
    /// pairs, averaged across outputs for multi-output regression. 1.0 is a perfect
    /// fit, 0.0 is no better than always predicting the mean. (Regression metric.)
    [[nodiscard]] double R2(const float* states, const float* targets,
                            size_t num_samples) const;

    /// @brief Classification accuracy over @p num_samples (state, label) pairs —
    /// the fraction predicted correctly. Labels are integer class indices.
    /// Multi-class compares argmax to the label; a single output is thresholded
    /// at 0. (Classification metric.)
    [[nodiscard]] double Accuracy(const float* states, const int* labels,
                                  size_t num_samples) const;

    // ----- Accessors -----

    /// @brief Size of one prediction: regression targets, or number of classes.
    [[nodiscard]] size_t NumOutputs() const { return num_outputs_; }
    /// @brief Length of the input field the network expects, N = 2^dim.
    [[nodiscard]] size_t NumFeatures() const { return num_features_; }
    /// @brief True after @ref Train starts (wrong-task throws first), after
    /// any @ref TrainStep / @ref TrainStepBatch, or after a non-empty
    /// @ref SetState / @ref LoadHcnnModel. False on a freshly constructed
    /// (random-init) net. An empty @ref SetState is ignored and does not
    /// flip this.
    [[nodiscard]] bool IsTrained() const { return trained_; }
    [[nodiscard]] const ReadoutConfig& GetConfig() const { return config_; }

    /// @brief 1-based epoch that produced the restored best weights after the last
    /// @ref Train with @c restore_best_epoch, or 0 if that path was not used / no
    /// snapshot was taken.
    [[nodiscard]] int BestEpoch() const { return best_epoch_; }

    // ----- Serialization -----

    /// @brief Snapshot the live CNN weights as an opaque blob, returned by value so
    /// the copy can't go stale behind a later TrainStep* call (streaming training
    /// mutates the network in place). Pair with @ref SetState.
    ///
    /// Format is an **unversioned** float32 layout promoted to double (same order as
    /// `HCNN::GetWeights`). Prefer @ref SaveHcnnModel for portable, versioned files.
    [[nodiscard]] std::vector<double> Weights() const;

    /// @brief Load a weight blob from @ref Weights back into the network.
    /// An empty blob is ignored.
    /// @param mode Eval (default) restores parameters only; ResumeTrain also resets
    ///        optimizer moments for clean continued training.
    void SetState(std::vector<double> weights,
                  ReadoutLoadMode mode = ReadoutLoadMode::Eval);

    /// Arch sidecar `format` token written by @ref SaveHcnnModel (`.arch.json`).
    static constexpr const char* kArchSidecarFormat =
        "hypercube_cascade_readout_arch";
    /// Arch sidecar format version written by @ref SaveHcnnModel (`.arch.json`).
    static constexpr int kArchSidecarVersion = 1;

    /// @brief Write HypercubeCNN-native weights + arch sidecar:
    ///   `@p path_stem.hcnw`      — versioned HCNW (via `hcnn::save_weights`)
    ///   `@p path_stem.arch.json` — architecture knobs + expanded layer list
    /// A trailing `.hcnw` or `.arch.json` on @p path_stem is stripped.
    void SaveHcnnModel(const std::string& path_stem) const;

    /// @brief Load `@p path_stem.hcnw` into this readout after validating
    /// `@p path_stem.arch.json` against the live architecture (when the sidecar
    /// exists). If the sidecar is missing, HCNW's own dim/task/layer checks still
    /// apply. @p mode follows @ref SetState.
    void LoadHcnnModel(const std::string& path_stem,
                       ReadoutLoadMode mode = ReadoutLoadMode::Eval);

    /// @brief Human-readable architecture + parameter count (for logs / demos).
    [[nodiscard]] std::string ArchSummary() const;

private:
    std::unique_ptr<hcnn::HCNN> net_;
    ReadoutConfig config_;
    size_t num_features_ = 0;
    size_t num_outputs_ = 1;
    int best_epoch_ = 0; ///< See @ref BestEpoch.
    bool trained_ = false; ///< See @ref IsTrained.

    void build_architecture();
};
