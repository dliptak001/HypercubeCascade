// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 David Liptak

#include "HCNN.h"
#include "HCNNNetwork.h"

#include <algorithm>
#include <cstring>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>

namespace hcnn {

HCNN::HCNN(int start_dim, int num_outputs, int input_channels,
           TaskType task_type, size_t num_threads)
    : net_(std::make_unique<HCNNNetwork>(start_dim, num_outputs, input_channels,
                                         task_type, num_threads)) {}

HCNN::~HCNN() = default;

// Defined in .cpp so HCNNNetwork is complete (same as destructor).
// Move only transfers unique_ptr + scratch; ThreadPool stays on the heap.
HCNN::HCNN(HCNN&&) noexcept = default;
HCNN& HCNN::operator=(HCNN&&) noexcept = default;

// ---------------------------------------------------------------------------
//  Architecture
// ---------------------------------------------------------------------------
void HCNN::AddConv(int c_out, Activation activation,
                   bool use_bias, bool use_batchnorm) {
    net_->add_conv(c_out, activation, use_bias, use_batchnorm);
}

void HCNN::AddPool(PoolType type) {
    net_->add_pool(type);
}

void HCNN::RandomizeWeights(float scale, uint64_t seed) {
    net_->randomize_all_weights(scale, seed);
}

// ---------------------------------------------------------------------------
//  Mode / optimizer
// ---------------------------------------------------------------------------
void HCNN::SetTraining(bool training) {
    net_->set_training(training);
}

void HCNN::SetOptimizer(OptimizerType type, float beta1, float beta2, float eps) {
    net_->set_optimizer(type, beta1, beta2, eps);
}

void HCNN::SetTrainDefaults(const TrainParams& params) {
    train_defaults_ = params;
}

const TrainParams& HCNN::GetTrainDefaults() const {
    return train_defaults_;
}

void HCNN::PrepareBuffers() {
    net_->prepare_all_buffers();
}

// ---------------------------------------------------------------------------
//  Inference
// ---------------------------------------------------------------------------
void HCNN::Embed(const float* raw_input, int input_length,
                 float* embedded_out) const {
    net_->embed_input(raw_input, input_length, embedded_out);
}

void HCNN::Forward(const float* embedded, float* logits) const {
    net_->forward(embedded, logits);
}

void HCNN::ensure_predict_buffers_() const {
    const size_t cap = static_cast<size_t>(net_->get_input_channels()) *
                       static_cast<size_t>(net_->get_start_N());
    if (predict_embed_.size() < cap)
        predict_embed_.resize(cap);
    const size_t ko = static_cast<size_t>(net_->get_num_outputs());
    if (predict_logits_.size() < ko)
        predict_logits_.resize(ko);
}

void HCNN::Predict(const float* raw_input, int input_length,
                   float* outputs) const {
    ensure_predict_buffers_();
    net_->embed_input(raw_input, input_length, predict_embed_.data());
    net_->forward(predict_embed_.data(), outputs);
}

int HCNN::PredictClass(const float* raw_input, int input_length) const {
    if (net_->get_task_type() != TaskType::Classification) {
        throw std::logic_error(
            "HCNN::PredictClass: only valid for TaskType::Classification");
    }
    ensure_predict_buffers_();
    const int K = net_->get_num_outputs();
    Predict(raw_input, input_length, predict_logits_.data());
    int best = 0;
    float best_v = predict_logits_[0];
    for (int i = 1; i < K; ++i) {
        if (predict_logits_[static_cast<size_t>(i)] > best_v) {
            best_v = predict_logits_[static_cast<size_t>(i)];
            best = i;
        }
    }
    return best;
}

void HCNN::ForwardBatch(const float* flat_inputs, int input_length,
                        int batch_size, float* logits_out) {
    net_->forward_batch(flat_inputs, input_length, batch_size, logits_out);
}

void HCNN::require_input_view_(HCNNInputView in, const char* api) const {
    const int cap = net_->get_input_channels() * net_->get_start_N();
    try {
        in.require_capacity(cap);
    } catch (const std::invalid_argument& e) {
        throw std::invalid_argument(std::string(api) + ": " + e.what());
    }
}

void HCNN::Predict(HCNNInputView in, float* outputs) const {
    require_input_view_(in, "HCNN::Predict");
    if (in.count() != 1) {
        throw std::invalid_argument(
            "HCNN::Predict(HCNNInputView): count must be 1 (use ForwardBatch)");
    }
    Predict(in.sample(0), in.capacity(), outputs);
}

int HCNN::PredictClass(HCNNInputView in) const {
    require_input_view_(in, "HCNN::PredictClass");
    if (in.count() != 1) {
        throw std::invalid_argument(
            "HCNN::PredictClass(HCNNInputView): count must be 1");
    }
    return PredictClass(in.sample(0), in.capacity());
}

void HCNN::ForwardBatch(HCNNInputView in, float* logits_out) {
    require_input_view_(in, "HCNN::ForwardBatch");
    if (logits_out == nullptr && in.count() > 0) {
        throw std::invalid_argument("HCNN::ForwardBatch: logits_out is null");
    }
    ForwardBatch(in.data(), in.capacity(), in.count(), logits_out);
}

// ---------------------------------------------------------------------------
//  Training — shared epoch driver
// ---------------------------------------------------------------------------
template <typename GatherTargets, typename TrainChunk>
void HCNN::train_epoch_impl_(const float* flat_inputs, int input_length,
                             int sample_count, int batch_size,
                             unsigned shuffle_seed,
                             GatherTargets&& gather_targets,
                             TrainChunk&& train_chunk) {
    if (batch_size <= 0) {
        throw std::invalid_argument("HCNN::TrainEpoch*: batch_size must be > 0");
    }
    if (sample_count < 0) {
        throw std::invalid_argument("HCNN::TrainEpoch*: sample_count must be >= 0");
    }
    if (sample_count == 0) return;

    const auto n  = static_cast<size_t>(sample_count);
    const auto il = static_cast<size_t>(input_length);
    const auto bs = static_cast<size_t>(batch_size);

    if (shuffle_seed != 0) {
        if (shuffle_idx_.size() < n) shuffle_idx_.resize(n);
        std::iota(shuffle_idx_.begin(),
                  shuffle_idx_.begin() + static_cast<std::ptrdiff_t>(n), 0);
        std::mt19937 rng(shuffle_seed);
        std::shuffle(shuffle_idx_.begin(),
                     shuffle_idx_.begin() + static_cast<std::ptrdiff_t>(n), rng);
        if (shuffle_inputs_.size() < bs * il)
            shuffle_inputs_.resize(bs * il);
    }

    for (int start = 0; start < sample_count; start += batch_size) {
        const int chunk = std::min(batch_size, sample_count - start);

        if (shuffle_seed != 0) {
            for (int i = 0; i < chunk; ++i) {
                const int j = shuffle_idx_[static_cast<size_t>(start + i)];
                std::memcpy(shuffle_inputs_.data() + static_cast<size_t>(i) * il,
                            flat_inputs + static_cast<size_t>(j) * il,
                            il * sizeof(float));
                gather_targets(i, j);
            }
            train_chunk(shuffle_inputs_.data(), chunk, start, /*shuffled=*/true);
        } else {
            train_chunk(flat_inputs + static_cast<size_t>(start) * il,
                        chunk, start, /*shuffled=*/false);
        }
    }
}

// ---------------------------------------------------------------------------
//  Training — classification (int / const int* targets)
// ---------------------------------------------------------------------------
void HCNN::TrainStep(const float* raw_input, int input_length, int target_class,
                     float learning_rate, float momentum, float weight_decay,
                     const float* class_weights) {
    net_->train_step(raw_input, input_length, target_class, learning_rate,
                     momentum, weight_decay, class_weights);
}

void HCNN::TrainBatch(const float* flat_inputs, int input_length,
                      const int* targets, int batch_size,
                      float learning_rate, float momentum, float weight_decay,
                      const float* class_weights) {
    net_->train_batch(flat_inputs, input_length, targets, batch_size,
                      learning_rate, momentum, weight_decay, class_weights);
}

void HCNN::TrainEpoch(const float* flat_inputs, int input_length,
                      const int* targets, int sample_count, int batch_size,
                      float learning_rate, float momentum, float weight_decay,
                      const float* class_weights, unsigned shuffle_seed) {
    if (shuffle_seed != 0) {
        const auto bs = static_cast<size_t>(batch_size);
        if (shuffle_targets_.size() < bs) shuffle_targets_.resize(bs);
    }

    train_epoch_impl_(
        flat_inputs, input_length, sample_count, batch_size, shuffle_seed,
        [&](int chunk_i, int sample_j) {
            shuffle_targets_[static_cast<size_t>(chunk_i)] = targets[sample_j];
        },
        [&](const float* inputs, int chunk, int start, bool shuffled) {
            const int* tgt = shuffled ? shuffle_targets_.data()
                                      : (targets + start);
            net_->train_batch(inputs, input_length, tgt, chunk,
                              learning_rate, momentum, weight_decay,
                              class_weights);
        });
}

void HCNN::TrainStep(const float* raw_input, int input_length, int target_class,
                     const TrainParams& params) {
    TrainStep(raw_input, input_length, target_class, params.learning_rate,
              params.momentum, params.weight_decay, params.class_weights);
}

void HCNN::TrainBatch(const float* flat_inputs, int input_length,
                      const int* targets, int batch_size,
                      const TrainParams& params) {
    TrainBatch(flat_inputs, input_length, targets, batch_size,
               params.learning_rate, params.momentum, params.weight_decay,
               params.class_weights);
}

void HCNN::TrainEpoch(const float* flat_inputs, int input_length,
                      const int* targets, int sample_count, int batch_size,
                      const TrainParams& params) {
    TrainEpoch(flat_inputs, input_length, targets, sample_count, batch_size,
               params.learning_rate, params.momentum, params.weight_decay,
               params.class_weights, params.shuffle_seed);
}

void HCNN::TrainStep(HCNNInputView in, int target_class,
                     const TrainParams& params) {
    require_input_view_(in, "HCNN::TrainStep");
    if (in.count() != 1) {
        throw std::invalid_argument(
            "HCNN::TrainStep(HCNNInputView): count must be 1 (use TrainBatch/Epoch)");
    }
    TrainStep(in.sample(0), in.capacity(), target_class, params);
}

void HCNN::TrainBatch(HCNNInputView in, const int* targets, int batch_size,
                      const TrainParams& params) {
    require_input_view_(in, "HCNN::TrainBatch");
    if (batch_size != in.count()) {
        throw std::invalid_argument(
            "HCNN::TrainBatch(HCNNInputView): batch_size must equal in.count()");
    }
    if (targets == nullptr && in.count() > 0) {
        throw std::invalid_argument("HCNN::TrainBatch: targets is null");
    }
    TrainBatch(in.data(), in.capacity(), targets, batch_size, params);
}

void HCNN::TrainEpoch(HCNNInputView in, const int* targets, int batch_size,
                      const TrainParams& params) {
    require_input_view_(in, "HCNN::TrainEpoch");
    if (targets == nullptr && in.count() > 0) {
        throw std::invalid_argument("HCNN::TrainEpoch: targets is null");
    }
    TrainEpoch(in.data(), in.capacity(), targets, in.count(), batch_size, params);
}

void HCNN::TrainStep(const float* raw_input, int input_length, int target_class) {
    TrainStep(raw_input, input_length, target_class, train_defaults_);
}

void HCNN::TrainBatch(const float* flat_inputs, int input_length,
                      const int* targets, int batch_size) {
    TrainBatch(flat_inputs, input_length, targets, batch_size, train_defaults_);
}

void HCNN::TrainEpoch(const float* flat_inputs, int input_length,
                      const int* targets, int sample_count, int batch_size) {
    TrainEpoch(flat_inputs, input_length, targets, sample_count, batch_size,
               train_defaults_);
}

void HCNN::TrainStep(HCNNInputView in, int target_class) {
    TrainStep(in, target_class, train_defaults_);
}

void HCNN::TrainBatch(HCNNInputView in, const int* targets, int batch_size) {
    TrainBatch(in, targets, batch_size, train_defaults_);
}

void HCNN::TrainEpoch(HCNNInputView in, const int* targets, int batch_size) {
    TrainEpoch(in, targets, batch_size, train_defaults_);
}

// ---------------------------------------------------------------------------
//  Training — regression (const float* targets; same Train* names)
// ---------------------------------------------------------------------------
void HCNN::TrainStep(const float* raw_input, int input_length,
                     const float* target, float learning_rate, float momentum,
                     float weight_decay) {
    net_->train_step_regression(raw_input, input_length, target, learning_rate,
                                momentum, weight_decay);
}

void HCNN::TrainBatch(const float* flat_inputs, int input_length,
                      const float* flat_targets, int batch_size,
                      float learning_rate, float momentum, float weight_decay) {
    net_->train_batch_regression(flat_inputs, input_length, flat_targets,
                                 batch_size, learning_rate, momentum,
                                 weight_decay);
}

void HCNN::TrainEpoch(const float* flat_inputs, int input_length,
                      const float* flat_targets, int sample_count,
                      int batch_size, float learning_rate, float momentum,
                      float weight_decay, unsigned shuffle_seed) {
    const auto K  = static_cast<size_t>(net_->get_num_outputs());
    const auto bs = static_cast<size_t>(batch_size);

    if (shuffle_seed != 0) {
        if (shuffle_targets_f_.size() < bs * K)
            shuffle_targets_f_.resize(bs * K);
    }

    train_epoch_impl_(
        flat_inputs, input_length, sample_count, batch_size, shuffle_seed,
        [&](int chunk_i, int sample_j) {
            std::memcpy(shuffle_targets_f_.data() + static_cast<size_t>(chunk_i) * K,
                        flat_targets + static_cast<size_t>(sample_j) * K,
                        K * sizeof(float));
        },
        [&](const float* inputs, int chunk, int start, bool shuffled) {
            const float* tgt = shuffled
                ? shuffle_targets_f_.data()
                : (flat_targets + static_cast<size_t>(start) * K);
            net_->train_batch_regression(inputs, input_length, tgt, chunk,
                                         learning_rate, momentum, weight_decay);
        });
}

void HCNN::TrainStep(const float* raw_input, int input_length,
                     const float* target, const TrainParams& params) {
    TrainStep(raw_input, input_length, target, params.learning_rate,
              params.momentum, params.weight_decay);
}

void HCNN::TrainBatch(const float* flat_inputs, int input_length,
                      const float* flat_targets, int batch_size,
                      const TrainParams& params) {
    TrainBatch(flat_inputs, input_length, flat_targets, batch_size,
               params.learning_rate, params.momentum, params.weight_decay);
}

void HCNN::TrainEpoch(const float* flat_inputs, int input_length,
                      const float* flat_targets, int sample_count,
                      int batch_size, const TrainParams& params) {
    TrainEpoch(flat_inputs, input_length, flat_targets, sample_count, batch_size,
               params.learning_rate, params.momentum, params.weight_decay,
               params.shuffle_seed);
}

void HCNN::TrainStep(HCNNInputView in, const float* target,
                     const TrainParams& params) {
    require_input_view_(in, "HCNN::TrainStep");
    if (in.count() != 1) {
        throw std::invalid_argument(
            "HCNN::TrainStep(HCNNInputView, float*): count must be 1");
    }
    TrainStep(in.sample(0), in.capacity(), target, params);
}

void HCNN::TrainBatch(HCNNInputView in, const float* flat_targets,
                      int batch_size, const TrainParams& params) {
    require_input_view_(in, "HCNN::TrainBatch");
    if (batch_size != in.count()) {
        throw std::invalid_argument(
            "HCNN::TrainBatch(HCNNInputView, float*): batch_size must equal "
            "in.count()");
    }
    TrainBatch(in.data(), in.capacity(), flat_targets, batch_size, params);
}

void HCNN::TrainEpoch(HCNNInputView in, const float* flat_targets,
                      int batch_size, const TrainParams& params) {
    require_input_view_(in, "HCNN::TrainEpoch");
    TrainEpoch(in.data(), in.capacity(), flat_targets, in.count(), batch_size,
               params);
}

void HCNN::TrainStep(const float* raw_input, int input_length,
                     const float* target) {
    TrainStep(raw_input, input_length, target, train_defaults_);
}

void HCNN::TrainBatch(const float* flat_inputs, int input_length,
                      const float* flat_targets, int batch_size) {
    TrainBatch(flat_inputs, input_length, flat_targets, batch_size,
               train_defaults_);
}

void HCNN::TrainEpoch(const float* flat_inputs, int input_length,
                      const float* flat_targets, int sample_count,
                      int batch_size) {
    TrainEpoch(flat_inputs, input_length, flat_targets, sample_count, batch_size,
               train_defaults_);
}

void HCNN::TrainStep(HCNNInputView in, const float* target) {
    TrainStep(in, target, train_defaults_);
}

void HCNN::TrainBatch(HCNNInputView in, const float* flat_targets,
                      int batch_size) {
    TrainBatch(in, flat_targets, batch_size, train_defaults_);
}

void HCNN::TrainEpoch(HCNNInputView in, const float* flat_targets,
                      int batch_size) {
    TrainEpoch(in, flat_targets, batch_size, train_defaults_);
}

// ---------------------------------------------------------------------------
//  Sizing accessors
// ---------------------------------------------------------------------------
int HCNN::GetStartDim() const       { return net_->get_start_dim(); }
int HCNN::GetStartN() const         { return net_->get_start_N(); }
int HCNN::GetCurrentDim() const     { return net_->get_current_dim(); }
int HCNN::GetInputChannels() const  { return net_->get_input_channels(); }
int HCNN::GetNumOutputs() const     { return net_->get_num_outputs(); }
size_t HCNN::GetNumConv() const     { return net_->get_num_conv(); }
size_t HCNN::GetNumPool() const     { return net_->get_num_pool(); }
TaskType HCNN::GetTaskType() const  { return net_->get_task_type(); }
OptimizerType HCNN::GetOptimizerType() const { return net_->get_optimizer_type(); }
bool HCNN::WeightsInitialized() const { return net_->weights_initialized(); }

void HCNN::require_weights_initialized_(const char* api) const {
    if (!net_->weights_initialized()) {
        throw std::logic_error(
            std::string(api) + ": call RandomizeWeights() first "
            "(weight blob requires a sized FLATTEN head)");
    }
}

// ---------------------------------------------------------------------------
//  Weight serialization
// ---------------------------------------------------------------------------

size_t HCNN::GetWeightCount() const {
    require_weights_initialized_("HCNN::GetWeightCount");
    size_t total = 0;
    for (size_t i = 0; i < net_->get_num_conv(); ++i) {
        const auto& conv = net_->get_conv(i);
        total += static_cast<size_t>(conv.get_kernel_size());
        total += static_cast<size_t>(conv.get_bias_size());
        if (conv.has_batchnorm()) {
            const size_t p = static_cast<size_t>(conv.get_bn_param_size());
            total += 4 * p;  // gamma, beta, running_mean, running_var
        }
    }
    const auto& ro = net_->get_readout();
    total += static_cast<size_t>(ro.get_weight_size());
    total += static_cast<size_t>(ro.get_bias_size());
    return total;
}

void HCNN::GetWeights(float* out, size_t n) const {
    require_weights_initialized_("HCNN::GetWeights");
    if (out == nullptr)
        throw std::invalid_argument("HCNN::GetWeights: out is null");
    const size_t need = GetWeightCount();
    if (n != need) {
        throw std::invalid_argument(
            "HCNN::GetWeights: n=" + std::to_string(n)
            + " != weight count " + std::to_string(need));
    }

    size_t offset = 0;
    for (size_t i = 0; i < net_->get_num_conv(); ++i) {
        const auto& conv = net_->get_conv(i);
        const int ks = conv.get_kernel_size();
        std::memcpy(out + offset, conv.get_kernel_data(),
                    static_cast<size_t>(ks) * sizeof(float));
        offset += static_cast<size_t>(ks);
        const int bs = conv.get_bias_size();
        if (bs > 0) {
            std::memcpy(out + offset, conv.get_bias_data(),
                        static_cast<size_t>(bs) * sizeof(float));
            offset += static_cast<size_t>(bs);
        }
        if (conv.has_batchnorm()) {
            const int p = conv.get_bn_param_size();
            const size_t bytes = static_cast<size_t>(p) * sizeof(float);
            std::memcpy(out + offset, conv.get_bn_gamma_data(), bytes);
            offset += static_cast<size_t>(p);
            std::memcpy(out + offset, conv.get_bn_beta_data(), bytes);
            offset += static_cast<size_t>(p);
            std::memcpy(out + offset, conv.get_bn_running_mean_data(), bytes);
            offset += static_cast<size_t>(p);
            std::memcpy(out + offset, conv.get_bn_running_var_data(), bytes);
            offset += static_cast<size_t>(p);
        }
    }

    const auto& ro = net_->get_readout();
    const int ws = ro.get_weight_size();
    std::memcpy(out + offset, ro.get_weight_data(),
                static_cast<size_t>(ws) * sizeof(float));
    offset += static_cast<size_t>(ws);
    const int rbs = ro.get_bias_size();
    std::memcpy(out + offset, ro.get_bias_data(),
                static_cast<size_t>(rbs) * sizeof(float));
    offset += static_cast<size_t>(rbs);

    if (offset != need) {
        throw std::logic_error(
            "HCNN::GetWeights: internal layout mismatch (offset "
            + std::to_string(offset) + " vs need " + std::to_string(need) + ")");
    }
}

std::vector<float> HCNN::GetWeights() const {
    const size_t n = GetWeightCount();
    std::vector<float> blob(n);
    GetWeights(blob.data(), n);
    return blob;
}

void HCNN::SetWeights(const float* data, size_t n, bool reset_optimizer_moments) {
    require_weights_initialized_("HCNN::SetWeights");
    if (data == nullptr)
        throw std::invalid_argument("HCNN::SetWeights: data is null");
    const size_t need = GetWeightCount();
    if (n != need) {
        throw std::invalid_argument(
            "HCNN::SetWeights: n=" + std::to_string(n)
            + " != weight count " + std::to_string(need));
    }

    size_t offset = 0;
    for (size_t i = 0; i < net_->get_num_conv(); ++i) {
        auto& conv = net_->get_conv(i);
        const int ks = conv.get_kernel_size();
        std::memcpy(conv.get_kernel_data(), data + offset,
                    static_cast<size_t>(ks) * sizeof(float));
        offset += static_cast<size_t>(ks);
        const int bs = conv.get_bias_size();
        if (bs > 0) {
            std::memcpy(conv.get_bias_data(), data + offset,
                        static_cast<size_t>(bs) * sizeof(float));
            offset += static_cast<size_t>(bs);
        }
        if (conv.has_batchnorm()) {
            const int p = conv.get_bn_param_size();
            const size_t bytes = static_cast<size_t>(p) * sizeof(float);
            std::memcpy(conv.get_bn_gamma_data(), data + offset, bytes);
            offset += static_cast<size_t>(p);
            std::memcpy(conv.get_bn_beta_data(), data + offset, bytes);
            offset += static_cast<size_t>(p);
            std::memcpy(conv.get_bn_running_mean_data(), data + offset, bytes);
            offset += static_cast<size_t>(p);
            std::memcpy(conv.get_bn_running_var_data(), data + offset, bytes);
            offset += static_cast<size_t>(p);
        }
    }

    auto& ro = net_->get_readout();
    const int ws = ro.get_weight_size();
    std::memcpy(ro.get_weight_data(), data + offset,
                static_cast<size_t>(ws) * sizeof(float));
    offset += static_cast<size_t>(ws);
    const int rbs = ro.get_bias_size();
    std::memcpy(ro.get_bias_data(), data + offset,
                static_cast<size_t>(rbs) * sizeof(float));
    offset += static_cast<size_t>(rbs);

    if (offset != need) {
        throw std::logic_error(
            "HCNN::SetWeights: internal layout mismatch (offset "
            + std::to_string(offset) + " vs need " + std::to_string(need) + ")");
    }

    if (reset_optimizer_moments)
        net_->reset_optimizer_moments();
}

void HCNN::SetWeights(const std::vector<float>& blob,
                      bool reset_optimizer_moments) {
    SetWeights(blob.data(), blob.size(), reset_optimizer_moments);
}

} // namespace hcnn
