// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 David Liptak

#include "HCNNTrainHelpers.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <numbers>
#include <stdexcept>
#include <string>
#include <vector>

namespace hcnn {

namespace {

// Portable little-endian I/O (ints + IEEE-754 binary32 floats).
static_assert(sizeof(float) == 4, "HCNW requires 32-bit float");

void write_u32_le(std::ostream& os, std::uint32_t v) {
    const unsigned char b[4] = {
        static_cast<unsigned char>(v & 0xFFu),
        static_cast<unsigned char>((v >> 8) & 0xFFu),
        static_cast<unsigned char>((v >> 16) & 0xFFu),
        static_cast<unsigned char>((v >> 24) & 0xFFu),
    };
    os.write(reinterpret_cast<const char*>(b), 4);
}

void write_i32_le(std::ostream& os, std::int32_t v) {
    write_u32_le(os, static_cast<std::uint32_t>(v));
}

void write_u64_le(std::ostream& os, std::uint64_t v) {
    write_u32_le(os, static_cast<std::uint32_t>(v & 0xFFFFFFFFu));
    write_u32_le(os, static_cast<std::uint32_t>((v >> 32) & 0xFFFFFFFFu));
}

void write_f32_le(std::ostream& os, float f) {
    std::uint32_t u = 0;
    std::memcpy(&u, &f, sizeof(u));
    write_u32_le(os, u);
}

void write_f32_le_array(std::ostream& os, const float* data, size_t n) {
    for (size_t i = 0; i < n; ++i)
        write_f32_le(os, data[i]);
}

std::uint32_t read_u32_le(std::istream& is) {
    unsigned char b[4];
    is.read(reinterpret_cast<char*>(b), 4);
    if (!is)
        throw std::runtime_error("hcnn::load_weights: short read (u32)");
    return static_cast<std::uint32_t>(b[0])
         | (static_cast<std::uint32_t>(b[1]) << 8)
         | (static_cast<std::uint32_t>(b[2]) << 16)
         | (static_cast<std::uint32_t>(b[3]) << 24);
}

std::int32_t read_i32_le(std::istream& is) {
    return static_cast<std::int32_t>(read_u32_le(is));
}

std::uint64_t read_u64_le(std::istream& is) {
    const std::uint64_t lo = read_u32_le(is);
    const std::uint64_t hi = read_u32_le(is);
    return lo | (hi << 32);
}

float read_f32_le(std::istream& is) {
    const std::uint32_t u = read_u32_le(is);
    float f = 0.0f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

void read_f32_le_array(std::istream& is, float* data, size_t n) {
    for (size_t i = 0; i < n; ++i)
        data[i] = read_f32_le(is);
}

} // namespace

// -----------------------------------------------------------------------------
// Metrics
// -----------------------------------------------------------------------------

int argmax(const float* v, int n) {
    if (v == nullptr || n <= 0)
        throw std::invalid_argument("hcnn::argmax: need non-null v and n > 0");
    int best = 0;
    for (int i = 1; i < n; ++i)
        if (v[i] > v[best]) best = i;
    return best;
}

float softmax_cross_entropy(const float* logits, int num_classes, int target) {
    if (logits == nullptr || num_classes <= 0)
        throw std::invalid_argument(
            "hcnn::softmax_cross_entropy: need non-null logits and num_classes > 0");
    if (target < 0 || target >= num_classes)
        throw std::invalid_argument(
            "hcnn::softmax_cross_entropy: target out of range");

    double max_l = logits[0];
    for (int i = 1; i < num_classes; ++i)
        if (logits[i] > max_l) max_l = logits[i];

    double sum_exp = 0.0;
    for (int i = 0; i < num_classes; ++i)
        sum_exp += std::exp(static_cast<double>(logits[i]) - max_l);

    return static_cast<float>(
        -(static_cast<double>(logits[target]) - max_l) + std::log(sum_exp));
}

HCNNClassEval evaluate_classification(HCNN& net,
                                      const float* flat_inputs,
                                      int input_length,
                                      const int* targets,
                                      int count) {
    if (flat_inputs == nullptr || targets == nullptr)
        throw std::invalid_argument(
            "hcnn::evaluate_classification: null inputs or targets");
    if (count <= 0)
        throw std::invalid_argument(
            "hcnn::evaluate_classification: count must be > 0");
    if (input_length <= 0)
        throw std::invalid_argument(
            "hcnn::evaluate_classification: input_length must be > 0");

    const int K = net.GetNumOutputs();
    std::vector<float> all_logits(static_cast<size_t>(count) * static_cast<size_t>(K));
    net.ForwardBatch(flat_inputs, input_length, count, all_logits.data());

    float total_loss = 0.0f;
    int correct = 0;
    for (int i = 0; i < count; ++i) {
        const float* logits = all_logits.data() + static_cast<size_t>(i) * K;
        total_loss += softmax_cross_entropy(logits, K, targets[i]);
        if (argmax(logits, K) == targets[i]) ++correct;
    }

    HCNNClassEval r;
    r.loss = total_loss / static_cast<float>(count);
    r.correct = correct;
    r.count = count;
    r.accuracy = 100.0f * static_cast<float>(correct) / static_cast<float>(count);
    return r;
}

HCNNClassEval evaluate_classification(HCNN& net, const HCNNFlatDataset& ds) {
    if (ds.count <= 0 || ds.input_length <= 0)
        throw std::invalid_argument(
            "hcnn::evaluate_classification: empty HCNNFlatDataset");

    // Public fields can drift from the vectors if callers mutate them by hand.
    // Size-check here so we never pass undersized buffers into ForwardBatch.
    const size_t need_in =
        static_cast<size_t>(ds.count) * static_cast<size_t>(ds.input_length);
    if (ds.inputs.size() < need_in)
        throw std::invalid_argument(
            "hcnn::evaluate_classification: inputs.size() < count * input_length");
    if (ds.targets.size() < static_cast<size_t>(ds.count))
        throw std::invalid_argument(
            "hcnn::evaluate_classification: targets.size() < count");

    return evaluate_classification(net,
                                   ds.inputs.data(),
                                   ds.input_length,
                                   ds.targets.data(),
                                   ds.count);
}

// -----------------------------------------------------------------------------
// Flat dataset
// -----------------------------------------------------------------------------

void HCNNFlatDataset::reset(int n, int len) {
    if (n < 0 || len < 0)
        throw std::invalid_argument(
            "HCNNFlatDataset::reset: n and len must be >= 0");
    // Build temps first so a throw leaves *this fully unchanged (strong guarantee).
    std::vector<float> new_inputs(
        static_cast<size_t>(n) * static_cast<size_t>(len));
    std::vector<int> new_targets(static_cast<size_t>(n));
    std::vector<float> empty_ft;
    inputs.swap(new_inputs);
    targets.swap(new_targets);
    float_targets.swap(empty_ft);
    count = n;
    input_length = len;
    num_outputs = 0;
}

void HCNNFlatDataset::reset_regression(int n, int len, int n_out) {
    if (n < 0 || len < 0 || n_out < 1)
        throw std::invalid_argument(
            "HCNNFlatDataset::reset_regression: n,len >= 0 and num_outputs >= 1");
    std::vector<float> new_inputs(
        static_cast<size_t>(n) * static_cast<size_t>(len));
    std::vector<float> new_ft(
        static_cast<size_t>(n) * static_cast<size_t>(n_out));
    std::vector<int> empty_cls;
    inputs.swap(new_inputs);
    float_targets.swap(new_ft);
    targets.swap(empty_cls);
    count = n;
    input_length = len;
    num_outputs = n_out;
}

HCNNRegEval evaluate_regression(HCNN& net, const HCNNFlatDataset& ds) {
    if (ds.count <= 0 || ds.input_length <= 0)
        throw std::invalid_argument(
            "hcnn::evaluate_regression: empty HCNNFlatDataset");
    if (!ds.has_float_targets())
        throw std::invalid_argument(
            "hcnn::evaluate_regression: dataset has no float_targets "
            "(use reset_regression)");

    const size_t need_in =
        static_cast<size_t>(ds.count) * static_cast<size_t>(ds.input_length);
    if (ds.inputs.size() < need_in)
        throw std::invalid_argument(
            "hcnn::evaluate_regression: inputs.size() < count * input_length");

    return evaluate_regression(net,
                               ds.inputs.data(),
                               ds.input_length,
                               ds.float_targets.data(),
                               ds.count,
                               ds.num_outputs);
}

// -----------------------------------------------------------------------------
// Cosine LR
// -----------------------------------------------------------------------------

float cosine_lr(float lr_max, float lr_min, int epoch, int num_epochs) {
    if (num_epochs <= 1)
        return lr_max;
    if (epoch < 0)
        epoch = 0;
    if (epoch >= num_epochs)
        epoch = num_epochs - 1;

    const float progress =
        static_cast<float>(epoch) / static_cast<float>(num_epochs - 1);
    return lr_min + 0.5f * (lr_max - lr_min)
        * (1.0f + std::cos(static_cast<float>(std::numbers::pi) * progress));
}

// -----------------------------------------------------------------------------
// Dual checkpoint
// -----------------------------------------------------------------------------

void HCNNDualCheckpoint::reset() {
    best_loss_weights_.clear();
    best_acc_weights_.clear();
    best_loss_ = std::numeric_limits<float>::infinity();
    best_loss_acc_ = -1.0f;
    best_loss_epoch_ = 0;
    best_acc_ = -1.0f;
    best_acc_loss_ = std::numeric_limits<float>::infinity();
    best_acc_epoch_ = 0;
}

HCNNDualCheckpointUpdate HCNNDualCheckpoint::observe(const HCNN& net,
                                                     float loss,
                                                     float accuracy,
                                                     int epoch) {
    HCNNDualCheckpointUpdate u;

    if (loss < best_loss_
        || (loss == best_loss_ && accuracy > best_loss_acc_)) {
        best_loss_ = loss;
        best_loss_acc_ = accuracy;
        best_loss_epoch_ = epoch;
        best_loss_weights_ = net.GetWeights();
        u.new_best_loss = true;
    }

    if (accuracy > best_acc_
        || (accuracy == best_acc_ && loss < best_acc_loss_)) {
        best_acc_ = accuracy;
        best_acc_loss_ = loss;
        best_acc_epoch_ = epoch;
        best_acc_weights_ = net.GetWeights();
        u.new_best_acc = true;
    }

    return u;
}

void HCNNDualCheckpoint::restore_best_loss(HCNN& net) const {
    if (best_loss_weights_.empty())
        throw std::logic_error(
            "HCNNDualCheckpoint::restore_best_loss: no best-loss snapshot");
    net.SetWeights(best_loss_weights_);
}

void HCNNDualCheckpoint::restore_best_acc(HCNN& net) const {
    if (best_acc_weights_.empty())
        throw std::logic_error(
            "HCNNDualCheckpoint::restore_best_acc: no best-acc snapshot");
    net.SetWeights(best_acc_weights_);
}

// -----------------------------------------------------------------------------
// Regression metrics
// -----------------------------------------------------------------------------

HCNNRegEval evaluate_regression(HCNN& net,
                                const float* flat_inputs,
                                int input_length,
                                const float* flat_targets,
                                int count,
                                int num_outputs) {
    if (flat_inputs == nullptr || flat_targets == nullptr)
        throw std::invalid_argument(
            "hcnn::evaluate_regression: null inputs or targets");
    if (count <= 0)
        throw std::invalid_argument(
            "hcnn::evaluate_regression: count must be > 0");
    if (input_length <= 0)
        throw std::invalid_argument(
            "hcnn::evaluate_regression: input_length must be > 0");

    int K = num_outputs;
    if (K <= 0)
        K = net.GetNumOutputs();
    if (K <= 0)
        throw std::invalid_argument(
            "hcnn::evaluate_regression: num_outputs must be > 0");

    std::vector<float> preds(
        static_cast<size_t>(count) * static_cast<size_t>(K));
    net.ForwardBatch(flat_inputs, input_length, count, preds.data());

    double mse_sum = 0.0;
    double tgt_sum = 0.0;
    double tgt_sq = 0.0;
    const int n_scalars = count * K;
    for (int i = 0; i < n_scalars; ++i) {
        const double t = flat_targets[i];
        const double d = static_cast<double>(preds[static_cast<size_t>(i)]) - t;
        mse_sum += d * d;
        tgt_sum += t;
        tgt_sq += t * t;
    }
    const double dn = static_cast<double>(n_scalars);
    HCNNRegEval r;
    r.mse = mse_sum / dn;
    r.target_var = tgt_sq / dn - (tgt_sum / dn) * (tgt_sum / dn);
    r.count = count;
    return r;
}

// -----------------------------------------------------------------------------
// Best-metric checkpoint
// -----------------------------------------------------------------------------

void HCNNBestMetricCheckpoint::reset() {
    weights_.clear();
    best_metric_ = std::numeric_limits<float>::infinity();
    best_epoch_ = 0;
}

bool HCNNBestMetricCheckpoint::observe(const HCNN& net, float metric, int epoch) {
    if (!(metric < best_metric_))
        return false;
    best_metric_ = metric;
    best_epoch_ = epoch;
    weights_ = net.GetWeights();
    return true;
}

void HCNNBestMetricCheckpoint::restore(HCNN& net) const {
    if (weights_.empty())
        throw std::logic_error(
            "HCNNBestMetricCheckpoint::restore: no snapshot");
    net.SetWeights(weights_);
}

// -----------------------------------------------------------------------------
// Versioned weight file I/O
// -----------------------------------------------------------------------------

void save_weights(const HCNN& net, const std::string& path) {
    if (!net.WeightsInitialized())
        throw std::logic_error(
            "hcnn::save_weights: call RandomizeWeights() first");
    if (path.empty())
        throw std::invalid_argument("hcnn::save_weights: empty path");

    const size_t n = net.GetWeightCount();
    std::vector<float> blob(n);
    net.GetWeights(blob.data(), n);

    std::ofstream os(path, std::ios::binary | std::ios::trunc);
    if (!os)
        throw std::runtime_error("hcnn::save_weights: cannot open " + path);

    os.write("HCNW", 4);
    write_u32_le(os, kHCNNWeightFileVersion);
    write_i32_le(os, net.GetStartDim());
    write_i32_le(os, net.GetCurrentDim());
    write_i32_le(os, net.GetNumOutputs());
    write_i32_le(os, net.GetInputChannels());
    write_i32_le(os, static_cast<std::int32_t>(net.GetTaskType()));
    write_i32_le(os, static_cast<std::int32_t>(net.GetNumConv()));
    write_i32_le(os, static_cast<std::int32_t>(net.GetNumPool()));
    write_u64_le(os, static_cast<std::uint64_t>(n));
    write_f32_le_array(os, blob.data(), n);
    if (!os)
        throw std::runtime_error("hcnn::save_weights: write failed for " + path);
}

void load_weights(HCNN& net, const std::string& path,
                  bool reset_optimizer_moments) {
    if (!net.WeightsInitialized())
        throw std::logic_error(
            "hcnn::load_weights: call RandomizeWeights() first "
            "(network must already match the saved architecture)");
    if (path.empty())
        throw std::invalid_argument("hcnn::load_weights: empty path");

    std::ifstream is(path, std::ios::binary);
    if (!is)
        throw std::runtime_error("hcnn::load_weights: cannot open " + path);

    char magic[4];
    is.read(magic, 4);
    if (!is || magic[0] != 'H' || magic[1] != 'C' || magic[2] != 'N'
        || magic[3] != 'W') {
        throw std::runtime_error(
            "hcnn::load_weights: bad magic (expected HCNW) in " + path);
    }

    const std::uint32_t version = read_u32_le(is);
    if (version != kHCNNWeightFileVersion) {
        throw std::runtime_error(
            "hcnn::load_weights: unsupported version "
            + std::to_string(version) + " (want "
            + std::to_string(kHCNNWeightFileVersion) + ") in " + path);
    }

    const int start_dim = read_i32_le(is);
    const int current_dim = read_i32_le(is);
    const int num_outputs = read_i32_le(is);
    const int input_channels = read_i32_le(is);
    const int task_type = read_i32_le(is);
    const int num_conv = read_i32_le(is);
    const int num_pool = read_i32_le(is);
    const std::uint64_t weight_count = read_u64_le(is);

    if (start_dim != net.GetStartDim()
        || current_dim != net.GetCurrentDim()
        || num_outputs != net.GetNumOutputs()
        || input_channels != net.GetInputChannels()
        || task_type != static_cast<int>(net.GetTaskType())
        || num_conv != static_cast<int>(net.GetNumConv())
        || num_pool != static_cast<int>(net.GetNumPool())) {
        throw std::runtime_error(
            "hcnn::load_weights: architecture mismatch vs live network in "
            + path);
    }

    const size_t need = net.GetWeightCount();
    if (weight_count != static_cast<std::uint64_t>(need)) {
        throw std::runtime_error(
            "hcnn::load_weights: weight_count "
            + std::to_string(weight_count) + " != GetWeightCount "
            + std::to_string(need) + " in " + path);
    }

    std::vector<float> blob(need);
    read_f32_le_array(is, blob.data(), need);
    if (!is)
        throw std::runtime_error(
            "hcnn::load_weights: short read of weight blob in " + path);

    net.SetWeights(blob.data(), need, reset_optimizer_moments);
}

} // namespace hcnn
