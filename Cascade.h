#pragma once

#include "CascadeConfig.h"

#include <cstddef>
#include <memory>
#include <span>
#include <vector>

/// @brief Exciter then Reservoir then Readout: length-N field → features → task.
///
/// ```
///   x[N] ──▶ Exciter ──▶ y[N] ──▶ * interstage_scale ──▶ Reservoir(T)
///        ──▶ z[N] ──▶ * readout_scale ──▶ Readout
/// ```
///
/// Lifecycle: @ref Collect → @ref TrainOnCollected → @ref Predict /
/// @ref PredictClass. @ref Run maps a field into @ref LastFeatures without
/// appending to the training set. Save/load, weight blobs, and
/// @ref Readout::IsTrained live on @ref readout(), not here.
///
/// Caller buffers are never mutated: @ref Exciter::ExciteCube scales in place,
/// so every public path copies into internal scratch first.
///
/// One instance is not thread-safe for concurrent public calls.
/// Parallelism is internal to bulk @ref CollectBatch / @ref Accuracy / @ref R2.
class Cascade
{
public:
    /// @brief Build Exciter, Reservoir, and Readout. Stamps @c cfg.dim onto
    ///        all three stages.
    /// @throws std::invalid_argument if @c dim is not in [5, 12],
    ///         @c exciter.subcube_dim is not in [1, dim], @c T is < 1,
    ///         @c interstage_scale or @c readout_scale is not finite and > 0,
    ///         or @c readout.num_outputs is < 1. Also throws the Exciter,
    ///         Reservoir, and Readout construction checks.
    explicit Cascade(const CascadeConfig& cfg);
    ~Cascade();

    Cascade(const Cascade&) = delete;
    Cascade& operator=(const Cascade&) = delete;
    Cascade(Cascade&&) = delete;
    Cascade& operator=(Cascade&&) = delete;

    // ----- Sizes / pieces -----

    [[nodiscard]] size_t Dim() const { return dim_; }
    [[nodiscard]] size_t N() const { return n_; }
    [[nodiscard]] size_t NumCollected() const { return num_collected_; }
    /// Collected feature rows, sample-major, length @ref NumCollected() * N.
    /// Empty when nothing has been collected. Valid until the next
    /// @ref Collect / @ref CollectBatch / @ref ClearCollected.
    [[nodiscard]] std::span<const float> CollectedFeatures() const
    {
        return collected_features_;
    }
    [[nodiscard]] size_t NumOutputs() const { return readout_->NumOutputs(); }
    /// Configured collect-thread preference (0 = auto). Actual workers used on
    /// a given bulk map are min(resolved, sample_count).
    [[nodiscard]] size_t CollectThreads() const { return cfg_.collect_threads; }

    /// Resolved knobs (@c dim stamped onto the three stages).
    [[nodiscard]] const CascadeConfig& config() const { return cfg_; }
    [[nodiscard]] const Exciter& exciter() const { return *exciter_; }
    [[nodiscard]] const Reservoir& reservoir() const { return *reservoir_; }
    /// The trainable head. Weights, HCNW save/load, IsTrained, ArchSummary.
    [[nodiscard]] Readout& readout() { return *readout_; }
    [[nodiscard]] const Readout& readout() const { return *readout_; }

    // ----- Map field → features -----

    /// Map one length-N field to features. Does not modify @p x.
    /// Updates @ref LastExciter, @ref LastInterstage, @ref LastReservoir,
    /// and @ref LastFeatures.
    /// @throws std::invalid_argument if @p x is not length N.
    void Run(std::span<const float> x);

    /// Exciter output from the most recent serial map (@ref Run, @ref Collect,
    /// @ref Predict, @ref PredictClass). Length N. Empty until the first
    /// serial map. Not updated by bulk @ref CollectBatch / @ref Accuracy /
    /// @ref R2. Valid until the next serial map.
    [[nodiscard]] std::span<const float> LastExciter() const { return last_exciter_; }

    /// Exciter output × interstage_scale from the most recent serial map.
    /// This is the field injected into the reservoir. Same lifetime as
    /// @ref LastExciter.
    [[nodiscard]] std::span<const float> LastInterstage() const
    {
        return last_interstage_;
    }

    /// Reservoir live output (before @c readout_scale) from the most recent
    /// serial map. Same lifetime as @ref LastExciter.
    [[nodiscard]] std::span<const float> LastReservoir() const
    {
        return last_reservoir_;
    }

    /// Features the Readout sees: Reservoir output × readout_scale.
    /// From the most recent completed map on this instance (@ref Run,
    /// @ref Collect, @ref Predict, @ref PredictClass, @ref Accuracy,
    /// @ref R2, or a batch collect). Valid until the next map on this
    /// instance; copy the values if they need to be kept.
    /// @ref ClearCollected does not clear this buffer.
    [[nodiscard]] std::span<const float> LastFeatures() const { return last_features_; }

    // ----- Collect / train -----

    /// Drop all samples collected for batch training.
    /// Does not free collect workers or the collect thread pool.
    void ClearCollected();

    /// Map @p x, append features + class label. Updates @ref LastFeatures.
    /// @throws std::invalid_argument if the task is Regression, @p x is not
    ///         length N, or @p class_label is not in [0, num_outputs).
    void Collect(std::span<const float> x, int class_label);

    /// Map @p x, append features + targets. @p target length must equal
    /// NumOutputs(). Updates @ref LastFeatures.
    /// @throws std::invalid_argument if the task is Classification, @p x is
    ///         not length N, or @p target is the wrong length.
    void Collect(std::span<const float> x, std::span<const float> target);

    /// Bulk collect (classification). @p fields_flat is sample-major, length
    /// count * N; @p labels length count. Independent maps fan across
    /// @ref CascadeConfig::collect_threads workers. Validates all labels before
    /// mapping. On success the last row is left in @ref LastFeatures. A throw
    /// leaves the collected set unchanged.
    /// @throws std::invalid_argument if the task is Regression, the buffers
    ///         do not match, or a label is out of range.
    void CollectBatch(std::span<const float> fields_flat,
                      std::span<const int> labels);

    /// Bulk collect (regression). @p targets_flat is sample-major, length
    /// count * NumOutputs(). Same worker fan-out as the classification
    /// overload. On success the last row is left in @ref LastFeatures. A throw
    /// leaves the collected set unchanged.
    /// @throws std::invalid_argument if the task is Classification or the
    ///         buffers do not match.
    void CollectBatch(std::span<const float> fields_flat,
                      std::span<const float> targets_flat);

    /// Batch-train the readout on all collected samples. Continues from
    /// the current readout weights. Does not clear the collected set.
    /// @throws std::invalid_argument if nothing has been collected.
    void TrainOnCollected();

    // ----- Inference -----

    /// Fresh map + readout forward; returns NumOutputs() floats.
    /// Updates @ref LastFeatures.
    /// @throws std::invalid_argument if @p x is not length N.
    [[nodiscard]] std::vector<float> Predict(std::span<const float> x);

    /// Fresh map + argmax class (classification only).
    /// Updates @ref LastFeatures.
    /// @throws std::invalid_argument if the task is Regression or @p x is
    ///         not length N.
    [[nodiscard]] int PredictClass(std::span<const float> x);

    /// Accuracy on the collected (training) set — not a test-set metric.
    /// @throws std::invalid_argument if the set is empty or the task is
    ///         Regression.
    [[nodiscard]] double AccuracyOnCollected() const;

    /// R² on the collected (training) set — not a test-set metric.
    /// @throws std::invalid_argument if the set is empty or the task is
    ///         Classification.
    [[nodiscard]] double R2OnCollected() const;

    /// Fresh map + classification accuracy on a caller-owned set.
    /// @p fields_flat is sample-major, length count * N; @p labels length
    /// count. Updates @ref LastFeatures to the last sample.
    /// @throws std::invalid_argument if the set is empty, the task is
    ///         Regression, the buffers do not match, or a label is out of
    ///         range.
    [[nodiscard]] double Accuracy(std::span<const float> fields_flat,
                                  std::span<const int> labels);

    /// Fresh map + R² on a caller-owned set.
    /// @p targets_flat is sample-major, length count * NumOutputs().
    /// Updates @ref LastFeatures to the last sample.
    /// @throws std::invalid_argument if the set is empty, the task is
    ///         Classification, or the buffers do not match.
    [[nodiscard]] double R2(std::span<const float> fields_flat,
                            std::span<const float> targets_flat);

private:
    struct CollectWorker
    {
        Exciter* ex = nullptr;
        Reservoir* res = nullptr;
        std::unique_ptr<Exciter> owned_ex;
        std::unique_ptr<Reservoir> owned_res;
        std::vector<float> field;
        std::vector<float> transit; ///< Exciter output, before interstage_scale.
        std::vector<float> excited;
        std::vector<float> drive;
    };

    struct CollectPool;

    void MapOn(CollectWorker& w, const float* x, float* dest) const;
    void MapInto(std::span<const float> x, float* dest);
    void MapBatchInto(std::span<const float> fields_flat,
                      std::vector<float>& out_features);
    void MapCollectedOne(std::span<const float> x);
    void MapFeaturesParallel(const float* fields_flat, size_t count,
                             float* dest);
    void RequireClassification() const;
    void RequireRegression() const;

    [[nodiscard]] size_t ResolveCollectThreads(size_t count) const;
    void EnsureCollectWorkers(size_t n);
    void EnsureCollectPool(size_t nthreads);
    void PublishLastRow(const float* rows, size_t count);

    CascadeConfig cfg_{};
    size_t dim_ = 0;
    size_t n_ = 0;

    std::unique_ptr<Exciter> exciter_;
    std::unique_ptr<Reservoir> reservoir_;
    std::unique_ptr<Readout> readout_;

    std::vector<float> s0_;
    std::vector<float> last_exciter_;
    std::vector<float> last_interstage_;
    std::vector<float> last_reservoir_;
    std::vector<float> last_features_;
    std::vector<float> eval_features_; ///< Accuracy / R2 remap scratch; grows, never shrinks.

    std::vector<float> collected_features_;
    std::vector<int> collected_labels_;
    std::vector<float> collected_targets_;
    size_t num_collected_ = 0;

    std::vector<CollectWorker> collect_workers_;
    std::unique_ptr<CollectPool> collect_pool_;
};
