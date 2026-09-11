// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 David Liptak

#pragma once

#include "HCNN.h"

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace hcnn {

// =============================================================================
// Optional training-loop helpers (not part of the conv/pool graph).
//
// Shared utilities for hosts and recipes that want a thin loop without
// re-implementing common pieces:
//   - classification metrics (softmax CE, argmax, batch evaluate)
//   - regression metrics (MSE, target variance, R^2)
//   - contiguous flat datasets (classification and/or regression targets)
//   - cosine LR schedule with floor
//   - dual weight checkpoints (best test loss + best test accuracy)
//   - best-metric checkpoint (minimize a scalar, e.g. test MSE)
//   - versioned weight file save/load (parameters only; rebuild arch first)
//
// Include this header only when you need these utilities.  HCNN itself does
// not depend on them; hosts with their own train loop can ignore this header.
// =============================================================================

// -----------------------------------------------------------------------------
// Classification metrics
// -----------------------------------------------------------------------------

/// Index of the maximum value in `v[0 .. n)`.  Requires n > 0.
[[nodiscard]] int argmax(const float* v, int n);

/// Numerically stable softmax cross-entropy for one sample.
/// `target` must be in [0, num_classes).
[[nodiscard]] float softmax_cross_entropy(const float* logits,
                                          int num_classes,
                                          int target);

/// Aggregate classification metrics over a dataset.
struct HCNNClassEval {
    float loss = 0.0f;       ///< Mean softmax CE
    float accuracy = 0.0f;   ///< Percent correct in [0, 100]
    int   correct = 0;
    int   count = 0;
};

/// Run `ForwardBatch` then compute mean CE and accuracy.
/// `flat_inputs` is `count * input_length` row-major; `targets` is `count` labels.
[[nodiscard]] HCNNClassEval evaluate_classification(
    HCNN& net,
    const float* flat_inputs,
    int input_length,
    const int* targets,
    int count);

// -----------------------------------------------------------------------------
// Flat dataset (classification and/or regression targets)
// -----------------------------------------------------------------------------

/**
 * Contiguous row-major buffers for HCNN train / infer APIs.
 *
 * Layout:
 *   - `inputs`:          count * input_length floats
 *   - `targets`:         count int class labels (classification; empty if unused)
 *   - `float_targets`:   count * num_outputs floats (regression; empty if unused)
 *
 * Use `reset` for classification-only, `reset_regression` for regression-only.
 * Both target buffers may be filled if you need dual-task bookkeeping, but
 * evaluate/train helpers pick one family based on which buffer is sized.
 */
struct HCNNFlatDataset {
    std::vector<float> inputs;
    std::vector<int>   targets;         ///< class indices; size count when used
    std::vector<float> float_targets;   ///< count * num_outputs when used
    int count = 0;
    int input_length = 0;
    int num_outputs = 0;  ///< regression target dim; 0 when classification-only

    /// Classification layout: size inputs + targets; clear float_targets.
    void reset(int n, int len);

    /// Regression layout: size inputs + float_targets; clear class targets.
    void reset_regression(int n, int len, int num_outputs);

    /// True when targets is sized for classification eval/train.
    [[nodiscard]] bool has_class_targets() const {
        return count > 0 && targets.size() >= static_cast<size_t>(count);
    }
    /// True when float_targets is sized for regression eval/train.
    [[nodiscard]] bool has_float_targets() const {
        return count > 0 && num_outputs > 0
            && float_targets.size()
                   >= static_cast<size_t>(count) * static_cast<size_t>(num_outputs);
    }

    /// Pointer to sample `i`'s input (length `input_length`).  No bounds check.
    [[nodiscard]] float* sample_input(int i) {
        return inputs.data() + static_cast<size_t>(i) * static_cast<size_t>(input_length);
    }
    [[nodiscard]] const float* sample_input(int i) const {
        return inputs.data() + static_cast<size_t>(i) * static_cast<size_t>(input_length);
    }

    /// Pointer to sample `i`'s regression target (length `num_outputs`).
    [[nodiscard]] float* sample_float_target(int i) {
        return float_targets.data()
            + static_cast<size_t>(i) * static_cast<size_t>(num_outputs);
    }
    [[nodiscard]] const float* sample_float_target(int i) const {
        return float_targets.data()
            + static_cast<size_t>(i) * static_cast<size_t>(num_outputs);
    }

    /**
     * Full-capacity view of `inputs` for typed HCNN overloads.
     * Requires `input_length` to already be the network capacity (e.g. N after
     * spatial embed).  Does not copy.
     */
    [[nodiscard]] HCNNInputView input_view() const {
        return HCNNInputView::from_full(inputs.data(), count, input_length);
    }
};

/// Convenience overload for HCNNFlatDataset (uses `targets`).
[[nodiscard]] HCNNClassEval evaluate_classification(HCNN& net,
                                                    const HCNNFlatDataset& ds);

// -----------------------------------------------------------------------------
// Regression metrics
// -----------------------------------------------------------------------------

/// Aggregate regression metrics over a dataset (MSE + R^2 ingredients).
struct HCNNRegEval {
    double mse = 0.0;          ///< Mean squared error over all target scalars
    double target_var = 0.0;   ///< Empirical variance of targets (population)
    int    count = 0;          ///< Number of samples (not target dims)

    /// Coefficient of determination. 1 = perfect; 0 = mean baseline; <0 worse.
    [[nodiscard]] double r2() const {
        return target_var > 0.0 ? 1.0 - mse / target_var : 0.0;
    }
};

/**
 * Run ForwardBatch then mean MSE over all output dimensions.
 * `flat_inputs` is count * input_length; `flat_targets` is count * num_outputs
 * (row-major).  When num_outputs is 0, uses net.GetNumOutputs().
 */
[[nodiscard]] HCNNRegEval evaluate_regression(
    HCNN& net,
    const float* flat_inputs,
    int input_length,
    const float* flat_targets,
    int count,
    int num_outputs = 0);

/// Convenience overload for HCNNFlatDataset (uses `float_targets` / `num_outputs`).
[[nodiscard]] HCNNRegEval evaluate_regression(HCNN& net, const HCNNFlatDataset& ds);

// -----------------------------------------------------------------------------
// Cosine LR schedule
// -----------------------------------------------------------------------------

/**
 * Cosine annealing from `lr_max` (epoch 0) down to `lr_min` (last epoch).
 *
 *   progress = epoch / max(num_epochs - 1, 1)   for epoch in [0, num_epochs)
 *   lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(pi * progress))
 *
 * When `num_epochs <= 1`, returns `lr_max`.  Negative epoch is treated as 0;
 * epoch >= num_epochs is clamped to the last step (lr_min when num_epochs > 1).
 *
 * Typical MNIST schedule: lr_max = 1e-3, lr_min = 1e-4 (10% floor).
 */
[[nodiscard]] float cosine_lr(float lr_max, float lr_min,
                              int epoch, int num_epochs);

// -----------------------------------------------------------------------------
// Thin training session helper (optional)
// -----------------------------------------------------------------------------

/**
 * Lightweight train loop helper: holds `TrainParams`, optional cosine LR, and
 * forwards to `HCNN` typed/flat train APIs.  Does not own the network.
 *
 * Typical use:
 * @code
 *   HCNNTrainer tr(net);
 *   tr.params().weight_decay = 1e-3f;
 *   tr.set_cosine(1e-3f, 1e-4f, 60);
 *   for (int e = 0; e < 60; ++e)
 *       tr.train_epoch(ds.input_view(), ds.targets.data(), batch, e);
 * @endcode
 *
 * Also syncs prepared params onto `HCNN::SetTrainDefaults` when
 * `sync_net_defaults` is true (default), so `net.TrainEpoch(...)` without
 * params uses the same knobs.
 */
class HCNNTrainer {
public:
    explicit HCNNTrainer(HCNN& net, bool sync_net_defaults = true)
        : net_(&net), sync_net_defaults_(sync_net_defaults) {
        params_ = net.GetTrainDefaults();
    }

    [[nodiscard]] HCNN& net() { return *net_; }
    [[nodiscard]] const HCNN& net() const { return *net_; }

    [[nodiscard]] TrainParams& params() { return params_; }
    [[nodiscard]] const TrainParams& params() const { return params_; }

    void set_params(const TrainParams& p) {
        params_ = p;
        if (sync_net_defaults_)
            net_->SetTrainDefaults(params_);
    }

    /// Cosine LR from lr_max (epoch 0) to lr_min (last epoch).  Also sets
    /// shuffle_seed = epoch + 1 in prepare_epoch.
    void set_cosine(float lr_max, float lr_min, int num_epochs) {
        if (num_epochs < 1) {
            throw std::invalid_argument(
                "HCNNTrainer::set_cosine: num_epochs must be >= 1");
        }
        lr_max_ = lr_max;
        lr_min_ = lr_min;
        num_epochs_ = num_epochs;
        use_cosine_ = true;
    }

    void clear_cosine() { use_cosine_ = false; }

    [[nodiscard]] bool has_cosine() const { return use_cosine_; }

    /// Update params.learning_rate (if cosine) and shuffle_seed = epoch+1.
    /// Optionally mirrors onto HCNN::SetTrainDefaults.
    void prepare_epoch(int epoch_0based) {
        if (use_cosine_) {
            params_.learning_rate =
                cosine_lr(lr_max_, lr_min_, epoch_0based, num_epochs_);
        }
        params_.shuffle_seed = static_cast<unsigned>(epoch_0based + 1);
        if (sync_net_defaults_)
            net_->SetTrainDefaults(params_);
    }

    // Classification (int labels)
    void train_epoch(HCNNInputView in, const int* targets, int batch_size,
                     int epoch_0based) {
        prepare_epoch(epoch_0based);
        net_->TrainEpoch(in, targets, batch_size, params_);
    }

    void train_epoch(const HCNNFlatDataset& ds, int batch_size,
                     int epoch_0based) {
        if (!ds.has_class_targets()) {
            throw std::invalid_argument(
                "HCNNTrainer::train_epoch: dataset has no class targets "
                "(use train_epoch with float targets / reset_regression)");
        }
        train_epoch(ds.input_view(), ds.targets.data(), batch_size, epoch_0based);
    }

    // Regression (float targets) — same train_epoch name
    void train_epoch(HCNNInputView in, const float* flat_targets, int batch_size,
                     int epoch_0based) {
        prepare_epoch(epoch_0based);
        net_->TrainEpoch(in, flat_targets, batch_size, params_);
    }

    /// Alias for regression FlatDataset (has float_targets).
    void train_epoch_regression(const HCNNFlatDataset& ds, int batch_size,
                                int epoch_0based) {
        if (!ds.has_float_targets()) {
            throw std::invalid_argument(
                "HCNNTrainer::train_epoch_regression: dataset has no float_targets");
        }
        train_epoch(ds.input_view(), ds.float_targets.data(), batch_size,
                    epoch_0based);
    }

    void train_epoch_regression(HCNNInputView in, const float* flat_targets,
                                int batch_size, int epoch_0based) {
        train_epoch(in, flat_targets, batch_size, epoch_0based);
    }

private:
    HCNN* net_ = nullptr;
    TrainParams params_;
    bool sync_net_defaults_ = true;
    bool use_cosine_ = false;
    float lr_max_ = 1e-3f;
    float lr_min_ = 1e-4f;
    int num_epochs_ = 1;
};

// -----------------------------------------------------------------------------
// Dual weight checkpoints
// -----------------------------------------------------------------------------

/// Flags returned by HCNNDualCheckpoint::observe.
struct HCNNDualCheckpointUpdate {
    bool new_best_loss = false;
    bool new_best_acc  = false;

    [[nodiscard]] bool any() const { return new_best_loss || new_best_acc; }
};

/**
 * Tracks two independent weight snapshots during training:
 *   - best (lowest) loss, with higher accuracy as tie-break
 *   - best (highest) accuracy, with lower loss as tie-break
 *
 * Uses HCNN::GetWeights / SetWeights.  The blob includes kernels, biases,
 * BN gamma/beta, and BN running mean/var when layers use batchnorm.
 * Optimizer moments and Adam timestep are **not** in the blob.
 *
 * Intended for **eval / export** of the best snapshot (as in the MNIST demo).
 * restore_* calls SetWeights(blob) with default reset_optimizer_moments=false.
 * To continue training after restore, call SetWeights(blob, true) or
 * SetOptimizer.
 */
class HCNNDualCheckpoint {
public:
    /// Record metrics for this epoch; copy weights if a new best is found.
    /// `epoch` is 1-based for reporting (matches typical "Epoch e/E" logs).
    HCNNDualCheckpointUpdate observe(const HCNN& net,
                                     float loss,
                                     float accuracy,
                                     int epoch);

    [[nodiscard]] bool has_best_loss() const { return !best_loss_weights_.empty(); }
    [[nodiscard]] bool has_best_acc()  const { return !best_acc_weights_.empty(); }

    void restore_best_loss(HCNN& net) const;
    void restore_best_acc(HCNN& net) const;

    [[nodiscard]] float best_loss() const { return best_loss_; }
    [[nodiscard]] float best_loss_acc() const { return best_loss_acc_; }
    [[nodiscard]] int   best_loss_epoch() const { return best_loss_epoch_; }

    [[nodiscard]] float best_acc() const { return best_acc_; }
    [[nodiscard]] float best_acc_loss() const { return best_acc_loss_; }
    [[nodiscard]] int   best_acc_epoch() const { return best_acc_epoch_; }

    /// Clear stored weights and reset metrics to initial sentinels.
    void reset();

private:
    std::vector<float> best_loss_weights_;
    std::vector<float> best_acc_weights_;

    float best_loss_ = std::numeric_limits<float>::infinity();
    float best_loss_acc_ = -1.0f;
    int   best_loss_epoch_ = 0;

    float best_acc_ = -1.0f;
    float best_acc_loss_ = std::numeric_limits<float>::infinity();
    int   best_acc_epoch_ = 0;
};

// -----------------------------------------------------------------------------
// Best-metric weight checkpoint (minimize)
// -----------------------------------------------------------------------------

/**
 * Tracks a single weight snapshot for the best (lowest) scalar metric so far.
 * Typical use: best test MSE in a regression loop.
 *
 * Same caveats as HCNNDualCheckpoint: blob is parameters + BN stats when
 * present; optimizer moments are not included.  Eval/export restore by default.
 */
class HCNNBestMetricCheckpoint {
public:
    /// Copy weights when metric improves (strictly lower).  epoch is 1-based.
    /// @return true if this observation set a new best.
    bool observe(const HCNN& net, float metric, int epoch);

    [[nodiscard]] bool has_best() const { return !weights_.empty(); }
    void restore(HCNN& net) const;

    [[nodiscard]] float best_metric() const { return best_metric_; }
    [[nodiscard]] int   best_epoch() const { return best_epoch_; }

    void reset();

private:
    std::vector<float> weights_;
    float best_metric_ = std::numeric_limits<float>::infinity();
    int   best_epoch_ = 0;
};

// -----------------------------------------------------------------------------
// Versioned weight file I/O
// -----------------------------------------------------------------------------

/// Current on-disk format version written by save_weights.
inline constexpr std::uint32_t kHCNNWeightFileVersion = 1;

/**
 * Write a versioned weight file for `net` (must be WeightsInitialized).
 *
 * Binary layout — **portable little-endian** on all hosts:
 *   magic[4] = 'H','C','N','W'
 *   uint32 version (= kHCNNWeightFileVersion)
 *   int32  start_dim, current_dim, num_outputs, input_channels
 *   int32  task_type (0=Classification, 1=Regression)
 *   int32  num_conv, num_pool
 *   uint64 weight_count
 *   float32 weights[weight_count]  // IEEE-754 binary32, little-endian
 *                                  // layout matches GetWeights
 *
 * Safe to ship across archs (x86_64, aarch64 LE, etc.).  Does not include
 * optimizer moments.  Throws on I/O failure.
 */
void save_weights(const HCNN& net, const std::string& path);

/**
 * Load weights from a file written by save_weights into an already-built net
 * (same architecture sizing).  Validates magic, version, dims, task, layer
 * counts, and weight_count against the live network.
 *
 * @param reset_optimizer_moments  forwarded to SetWeights (default false = eval).
 */
void load_weights(HCNN& net, const std::string& path,
                  bool reset_optimizer_moments = false);

} // namespace hcnn

