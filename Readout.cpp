#include "Readout.h"

#include "HCNN.h"
#include "HCNNArch.h"
#include "HCNNTrainHelpers.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <fstream>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>


// ---------------------------------------------------------------------------
//  Mapping helpers (HCNN stays out of the public header)
// ---------------------------------------------------------------------------

static hcnn::Activation map_activation(ReadoutActivation a)
{
    switch (a) {
        case ReadoutActivation::TANH:       return hcnn::Activation::TANH;
        case ReadoutActivation::RELU:       return hcnn::Activation::RELU;
        case ReadoutActivation::LEAKY_RELU: return hcnn::Activation::LEAKY_RELU;
        case ReadoutActivation::NONE:       return hcnn::Activation::NONE;
    }
    return hcnn::Activation::TANH;
}

static hcnn::PoolType map_pool(ReadoutPoolType t)
{
    switch (t) {
        case ReadoutPoolType::Max: return hcnn::PoolType::MAX;
        case ReadoutPoolType::Avg: return hcnn::PoolType::AVG;
    }
    return hcnn::PoolType::MAX;
}

static hcnn::OptimizerType map_optimizer(ReadoutOptimizer o)
{
    switch (o) {
        case ReadoutOptimizer::Adam: return hcnn::OptimizerType::ADAM;
        case ReadoutOptimizer::Sgd:  return hcnn::OptimizerType::SGD;
    }
    return hcnn::OptimizerType::ADAM;
}

/// Resolved Conv(+Pool) stage count. 0 in config means auto: min(dim-2, 2), at least 1.
static int resolved_num_layers(const ReadoutConfig& cfg)
{
    const int d = static_cast<int>(cfg.dim);
    int layers = (cfg.num_layers > 0) ? cfg.num_layers : std::min(d - 2, 2);
    return std::max(layers, 1);
}

/// Hard limits match HypercubeCNN ([3, 30]). Pooling must leave dim >= 2.
static void validate_readout_config(const ReadoutConfig& cfg)
{
    if (cfg.dim < 3 || cfg.dim > 30)
    {
        throw std::invalid_argument(
            "Readout: dim must be in 3 <= dim <= 30");
    }
    if (cfg.num_outputs < 1)
        throw std::invalid_argument("Readout: num_outputs must be >= 1");
    if (cfg.num_layers < 0)
        throw std::invalid_argument("Readout: num_layers must be >= 0");
    if (cfg.conv_channels < 1)
        throw std::invalid_argument("Readout: conv_channels must be >= 1");
    if (cfg.channel_growth < 1)
        throw std::invalid_argument("Readout: channel_growth must be >= 1");

    const int d = static_cast<int>(cfg.dim);
    const int layers = resolved_num_layers(cfg);
    // Each pool drops one hypercube dimension; HCNN requires current_dim >= 2
    // before a pool. With pooling off the dimension never shrinks.
    if (cfg.use_pooling && layers > d - 2)
    {
        throw std::invalid_argument(
            "Readout: with pooling, num_layers must be <= dim-2 "
            "(must leave dim >= 2 after the stack)");
    }
}

static std::vector<hcnn::LayerSpec> make_layer_specs(const ReadoutConfig& cfg)
{
    const int layers = resolved_num_layers(cfg);

    const hcnn::Activation act = map_activation(cfg.activation);
    const hcnn::PoolType pool = map_pool(cfg.pool_type);

    std::vector<hcnn::LayerSpec> specs;
    specs.reserve(static_cast<size_t>(layers) * (cfg.use_pooling ? 2u : 1u));

    int ch = cfg.conv_channels;
    for (int i = 0; i < layers; ++i) {
        specs.push_back(hcnn::LayerSpec::Conv(
            ch, act, /*bias=*/true, cfg.use_batchnorm));
        // Antipodal pool mixes every bit (including block-index bits). Conv-only
        // keeps vertex structure into the FLATTEN head.
        if (cfg.use_pooling)
            specs.push_back(hcnn::LayerSpec::Pool(pool));
        ch *= cfg.channel_growth;
    }
    return specs;
}

/// Hold-out size: 0 means score the full set (and train on all of it).
static void holdout_counts(const ReadoutConfig& cfg, size_t num_samples,
                           size_t& n_train, size_t& n_score)
{
    n_train = num_samples;
    n_score = num_samples;
    if (!cfg.restore_best_epoch || cfg.best_epoch_holdout_frac <= 0.0f
        || num_samples < 2)
        return;

    float frac = cfg.best_epoch_holdout_frac;
    if (frac < 0.0f) frac = 0.0f;
    if (frac > 0.5f) frac = 0.5f;
    size_t n_val = static_cast<size_t>(
        static_cast<float>(num_samples) * frac + 0.5f);
    if (n_val < 1) n_val = 1;
    if (n_val >= num_samples) n_val = num_samples / 2;
    n_train = num_samples - n_val;
    n_score = n_val;
}

static std::vector<size_t> shuffled_order(size_t n, uint64_t seed)
{
    std::vector<size_t> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    if (n > 1)
    {
        std::mt19937_64 rng(seed ^ 0x484F4C444F555400ULL);
        std::shuffle(idx.begin(), idx.end(), rng);
    }
    return idx;
}

static void gather_states(const float* src, float* dst,
                          const std::vector<size_t>& order, size_t n_feat)
{
    for (size_t i = 0; i < order.size(); ++i)
    {
        std::memcpy(dst + i * n_feat, src + order[i] * n_feat,
                    n_feat * sizeof(float));
    }
}

// ---------------------------------------------------------------------------
//  Lifecycle
// ---------------------------------------------------------------------------

Readout::Readout(const ReadoutConfig& cfg)
    : config_(cfg)
    , num_outputs_(static_cast<size_t>(cfg.num_outputs))
{
    validate_readout_config(config_);
    // Build the network eagerly. build_architecture() needs only the config
    // (no data, no warm-up), so there is nothing to defer: net_ is a non-null
    // invariant from construction on.
    num_features_ = 1ULL << config_.dim;
    build_architecture();
    net_->PrepareBuffers();
}

Readout::~Readout() = default;
Readout::Readout(Readout&&) noexcept = default;
Readout& Readout::operator=(Readout&&) noexcept = default;

void Readout::build_architecture()
{
    const int d = static_cast<int>(config_.dim);

    auto task_type = (config_.task == ReadoutTask::Classification)
                         ? hcnn::TaskType::Classification
                         : hcnn::TaskType::Regression;

    hcnn::HCNNConfig hcfg;
    hcfg.start_dim = d;
    hcfg.num_outputs = config_.num_outputs;
    hcfg.input_channels = 1;
    hcfg.task = task_type;
    hcfg.num_threads = config_.num_threads;
    hcfg.layers = make_layer_specs(config_);
    hcfg.optimizer = map_optimizer(config_.optimizer);
    hcfg.randomize = true;
    hcfg.weight_scale = 0.0f; // He/Xavier per layer
    hcfg.weight_seed = config_.seed;

    // Validate layer list + sizing before allocating (throws on bad stacks).
    (void)hcfg.summarize();
    net_ = hcfg.Build();
}

// ---------------------------------------------------------------------------
//  Training
// ---------------------------------------------------------------------------

void Readout::Train(const float* states, const float* targets,
                    size_t num_samples)
{
    if (config_.task != ReadoutTask::Regression)
        throw std::logic_error(
            "Readout::Train(float*): regression task only; use int* labels "
            "for classification");

    trained_ = true;
    // net_ is already built (ctor). Train fits the existing network in place;
    // a second Train() continues from the current weights rather than
    // re-randomizing — reconstruct the Readout for a fresh fit.
    const int n = static_cast<int>(num_features_);
    const size_t K = num_outputs_;
    best_epoch_ = 0;

    const float lr_min = config_.lr_max * config_.lr_min_frac;
    const int horizon = (config_.lr_decay_epochs > 0)
                            ? config_.lr_decay_epochs
                            : config_.epochs;

    size_t n_train = 0;
    size_t n_score = 0;
    holdout_counts(config_, num_samples, n_train, n_score);

    const float* train_states = states;
    const float* train_targets = targets;
    const float* score_states = states;
    const float* score_targets = targets;
    std::vector<float> shuf_x;
    std::vector<float> shuf_y;

    if (n_train != num_samples)
    {
        const auto order = shuffled_order(num_samples, config_.seed);
        const size_t nf = static_cast<size_t>(n);
        shuf_x.resize(num_samples * nf);
        shuf_y.resize(num_samples * K);
        gather_states(states, shuf_x.data(), order, nf);
        for (size_t i = 0; i < order.size(); ++i)
        {
            std::memcpy(shuf_y.data() + i * K, targets + order[i] * K,
                        K * sizeof(float));
        }
        train_states = shuf_x.data();
        train_targets = shuf_y.data();
        score_states = shuf_x.data() + n_train * nf;
        score_targets = shuf_y.data() + n_train * K;
    }

    const hcnn::HCNNInputView train_in = hcnn::HCNNInputView::from_full(
        train_states, static_cast<int>(n_train), n);

    hcnn::HCNNTrainer trainer(*net_);
    trainer.params().momentum = config_.momentum;
    trainer.params().weight_decay = config_.weight_decay;
    trainer.set_cosine(config_.lr_max, lr_min, std::max(horizon, 1));

    hcnn::HCNNBestMetricCheckpoint best_reg;

    for (int e = 0; e < config_.epochs; ++e) {
        trainer.train_epoch(train_in, train_targets, config_.batch_size, e);

        const bool want_tick = config_.epoch_tick != nullptr;
        const bool want_score = config_.restore_best_epoch && n_score > 0;
        const bool same_set = (train_states == score_states && n_train == n_score);

        if (want_tick && want_score && same_set)
        {
            hcnn::HCNNRegEval r = hcnn::evaluate_regression(
                *net_, train_states, n, train_targets,
                static_cast<int>(n_train), static_cast<int>(K));
            best_reg.observe(*net_, static_cast<float>(r.mse), e + 1);
            config_.epoch_tick(e + 1, r.mse);
            continue;
        }
        if (want_tick && n_train > 0)
        {
            hcnn::HCNNRegEval r = hcnn::evaluate_regression(
                *net_, train_states, n, train_targets,
                static_cast<int>(n_train), static_cast<int>(K));
            config_.epoch_tick(e + 1, r.mse);
        }
        if (want_score)
        {
            hcnn::HCNNRegEval r = hcnn::evaluate_regression(
                *net_, score_states, n, score_targets,
                static_cast<int>(n_score), static_cast<int>(K));
            best_reg.observe(*net_, static_cast<float>(r.mse), e + 1);
        }
    }

    if (!config_.restore_best_epoch)
        return;

    if (best_reg.has_best()) {
        best_reg.restore(*net_);
        best_epoch_ = best_reg.best_epoch();
    }
}

void Readout::Train(const float* states, const int* class_labels,
                    size_t num_samples)
{
    if (config_.task != ReadoutTask::Classification)
        throw std::logic_error(
            "Readout::Train(int*): classification task only; use float* "
            "targets for regression");

    trained_ = true;
    const int n = static_cast<int>(num_features_);
    best_epoch_ = 0;

    const float lr_min = config_.lr_max * config_.lr_min_frac;
    const int horizon = (config_.lr_decay_epochs > 0)
                            ? config_.lr_decay_epochs
                            : config_.epochs;

    size_t n_train = 0;
    size_t n_score = 0;
    holdout_counts(config_, num_samples, n_train, n_score);

    const float* train_states = states;
    const int* train_labels = class_labels;
    const float* score_states = states;
    const int* score_int = class_labels;
    std::vector<float> shuf_x;
    std::vector<int> shuf_y;

    if (n_train != num_samples)
    {
        const auto order = shuffled_order(num_samples, config_.seed);
        const size_t nf = static_cast<size_t>(n);
        shuf_x.resize(num_samples * nf);
        shuf_y.resize(num_samples);
        gather_states(states, shuf_x.data(), order, nf);
        for (size_t i = 0; i < order.size(); ++i)
            shuf_y[i] = class_labels[order[i]];
        train_states = shuf_x.data();
        train_labels = shuf_y.data();
        score_states = shuf_x.data() + n_train * nf;
        score_int = shuf_y.data() + n_train;
    }

    const hcnn::HCNNInputView train_in = hcnn::HCNNInputView::from_full(
        train_states, static_cast<int>(n_train), n);

    hcnn::HCNNTrainer trainer(*net_);
    trainer.params().momentum = config_.momentum;
    trainer.params().weight_decay = config_.weight_decay;
    trainer.set_cosine(config_.lr_max, lr_min, std::max(horizon, 1));

    hcnn::HCNNDualCheckpoint best_cls;

    for (int e = 0; e < config_.epochs; ++e) {
        trainer.train_epoch(train_in, train_labels, config_.batch_size, e);

        const bool want_tick = config_.epoch_tick != nullptr;
        const bool want_score = config_.restore_best_epoch && n_score > 0;
        const bool same_set = (train_states == score_states && n_train == n_score);

        if (want_tick && want_score && same_set)
        {
            hcnn::HCNNClassEval r = hcnn::evaluate_classification(
                *net_, train_states, n, train_labels,
                static_cast<int>(n_train));
            best_cls.observe(*net_, r.loss, r.accuracy, e + 1);
            config_.epoch_tick(e + 1, static_cast<double>(r.loss));
            continue;
        }
        if (want_tick && n_train > 0)
        {
            hcnn::HCNNClassEval r = hcnn::evaluate_classification(
                *net_, train_states, n, train_labels,
                static_cast<int>(n_train));
            config_.epoch_tick(e + 1, static_cast<double>(r.loss));
        }
        if (want_score)
        {
            hcnn::HCNNClassEval r = hcnn::evaluate_classification(
                *net_, score_states, n, score_int,
                static_cast<int>(n_score));
            best_cls.observe(*net_, r.loss, r.accuracy, e + 1);
        }
    }

    if (!config_.restore_best_epoch)
        return;

    if (best_cls.has_best_acc()) {
        best_cls.restore_best_acc(*net_);
        best_epoch_ = best_cls.best_acc_epoch();
    }
}

void Readout::TrainStep(const float* state, const float* target,
                        float lr, float weight_decay)
{
    if (config_.task != ReadoutTask::Regression)
        throw std::logic_error(
            "Readout::TrainStep(float*): regression only; use int class_label "
            "for classification");
    assert(net_);
    const int n = static_cast<int>(num_features_);

    hcnn::TrainParams p;
    p.learning_rate = lr;
    p.momentum = config_.momentum;
    p.weight_decay = weight_decay;
    net_->TrainStep(state, n, target, p);
    trained_ = true;
}

void Readout::TrainStep(const float* state, int class_label,
                        float lr, float weight_decay)
{
    if (config_.task != ReadoutTask::Classification)
        throw std::logic_error(
            "Readout::TrainStep(int): classification only; use float* target "
            "for regression");
    assert(net_);
    const int n = static_cast<int>(num_features_);

    hcnn::TrainParams p;
    p.learning_rate = lr;
    p.momentum = config_.momentum;
    p.weight_decay = weight_decay;
    net_->TrainStep(state, n, class_label, p);
    trained_ = true;
}

void Readout::TrainStepBatch(const float* states, const float* targets,
                             size_t count, float lr, float weight_decay)
{
    if (config_.task != ReadoutTask::Regression)
        throw std::logic_error(
            "Readout::TrainStepBatch(float*): regression only; use int* "
            "class_labels for classification");
    assert(net_);
    const int n = static_cast<int>(num_features_);
    const int batch = static_cast<int>(count);

    hcnn::TrainParams p;
    p.learning_rate = lr;
    p.momentum = config_.momentum;
    p.weight_decay = weight_decay;
    net_->TrainBatch(states, n, targets, batch, p);
    trained_ = true;
}

void Readout::TrainStepBatch(const float* states, const int* class_labels,
                             size_t count, float lr, float weight_decay)
{
    if (config_.task != ReadoutTask::Classification)
        throw std::logic_error(
            "Readout::TrainStepBatch(int*): classification only; use float* "
            "targets for regression");
    assert(net_);
    const int n = static_cast<int>(num_features_);
    const int batch = static_cast<int>(count);

    hcnn::TrainParams p;
    p.learning_rate = lr;
    p.momentum = config_.momentum;
    p.weight_decay = weight_decay;
    net_->TrainBatch(states, n, class_labels, batch, p);
    trained_ = true;
}

// ---------------------------------------------------------------------------
//  Prediction
// ---------------------------------------------------------------------------

void Readout::PredictRaw(const float* state, float* output) const
{
    assert(net_);
    const int n = static_cast<int>(num_features_);
    // HCNN::Predict embeds into internal scratch and writes raw logits / preds.
    net_->Predict(state, n, output);
}

int Readout::PredictClass(const float* state) const
{
    assert(net_);
    const int n = static_cast<int>(num_features_);
    return net_->PredictClass(state, n);
}

// ---------------------------------------------------------------------------
//  Evaluation
// ---------------------------------------------------------------------------

double Readout::R2(const float* states, const float* targets,
                   const size_t num_samples) const
{
    if (num_samples == 0) return 0.0;
    const int n = static_cast<int>(num_features_);
    const size_t K = num_outputs_;
    const int n_samples = static_cast<int>(num_samples);

    // Batch forward once (multi-output R²: average of per-output R² —
    // not HypercubeCNN's global MSE/variance helper).
    std::vector<float> preds(num_samples * K);
    net_->ForwardBatch(states, n, n_samples, preds.data());

    double r2_sum = 0.0;
    for (size_t k = 0; k < K; ++k) {
        double tgt_mean = 0.0;
        for (size_t s = 0; s < num_samples; ++s)
            tgt_mean += targets[s * K + k];
        tgt_mean /= static_cast<double>(num_samples);

        double ss_res = 0.0, ss_tot = 0.0;
        for (size_t s = 0; s < num_samples; ++s) {
            double y  = targets[s * K + k];
            double yh = preds[s * K + k];
            ss_res += (y - yh) * (y - yh);
            ss_tot += (y - tgt_mean) * (y - tgt_mean);
        }
        r2_sum += (ss_tot < 1e-12) ? 0.0 : (1.0 - ss_res / ss_tot);
    }
    return r2_sum / static_cast<double>(K);
}

double Readout::Accuracy(const float* states, const int* labels,
                         const size_t num_samples) const
{
    if (num_samples == 0) return 0.0;
    const int n = static_cast<int>(num_features_);
    const size_t K = num_outputs_;
    const int n_samples = static_cast<int>(num_samples);
    size_t correct = 0;

    if (K > 1) {
        std::vector<float> logits(num_samples * K);
        net_->ForwardBatch(states, n, n_samples, logits.data());
        for (size_t s = 0; s < num_samples; ++s) {
            const float* row = logits.data() + s * K;
            const int pred = static_cast<int>(
                std::max_element(row, row + K) - row);
            if (pred == labels[s]) ++correct;
        }
    } else {
        std::vector<float> preds(num_samples);
        net_->ForwardBatch(states, n, n_samples, preds.data());
        for (size_t s = 0; s < num_samples; ++s) {
            const int pred = (preds[s] > 0.0f) ? 1 : 0;
            if (pred == labels[s]) ++correct;
        }
    }
    return static_cast<double>(correct) / static_cast<double>(num_samples);
}

// ---------------------------------------------------------------------------
//  Serialization
// ---------------------------------------------------------------------------

std::vector<double> Readout::Weights() const
{
    // Snapshot the live network's weights on demand, by value — a returned copy
    // can't go stale behind a later TrainStep* call (streaming training mutates
    // net_ in place).
    const std::vector<float> fw = net_->GetWeights();
    return std::vector<double>(fw.begin(), fw.end());
}

void Readout::SetState(std::vector<double> weights, ReadoutLoadMode mode)
{
    // net_ is built (optimizer + buffers prepared) in the ctor — load the saved
    // weights straight into the existing, ready-to-train network.
    if (weights.empty()) return;
    const std::vector<float> fw(weights.begin(), weights.end());
    const bool reset_moments = (mode == ReadoutLoadMode::ResumeTrain);
    net_->SetWeights(fw, reset_moments);
    trained_ = true;
}

// ---------------------------------------------------------------------------
//  HypercubeCNN-native model I/O (HCNW + arch sidecar)
// ---------------------------------------------------------------------------

static const char* activation_token(ReadoutActivation a)
{
    switch (a) {
        case ReadoutActivation::TANH:       return "tanh";
        case ReadoutActivation::RELU:       return "relu";
        case ReadoutActivation::LEAKY_RELU: return "leaky_relu";
        case ReadoutActivation::NONE:       return "none";
    }
    return "tanh";
}

static const char* pool_token(ReadoutPoolType t)
{
    return (t == ReadoutPoolType::Avg) ? "avg" : "max";
}

static const char* optimizer_token(ReadoutOptimizer o)
{
    return (o == ReadoutOptimizer::Sgd) ? "sgd" : "adam";
}

static const char* hcnn_act_token(hcnn::Activation a)
{
    switch (a) {
        case hcnn::Activation::TANH:       return "tanh";
        case hcnn::Activation::RELU:       return "relu";
        case hcnn::Activation::LEAKY_RELU: return "leaky_relu";
        case hcnn::Activation::NONE:       return "none";
    }
    return "tanh";
}

static const char* hcnn_pool_token(hcnn::PoolType t)
{
    return (t == hcnn::PoolType::AVG) ? "avg" : "max";
}

/// Strip a trailing known extension so callers may pass either a stem or a path.
static std::string path_stem_normalized(std::string path)
{
    auto ends_with = [&](const char* ext) {
        const size_t n = std::char_traits<char>::length(ext);
        return path.size() >= n
            && path.compare(path.size() - n, n, ext) == 0;
    };
    if (ends_with(".hcnw"))
        path.resize(path.size() - 5);
    else if (ends_with(".arch.json"))
        path.resize(path.size() - 10);
    return path;
}

static bool file_exists(const std::string& path)
{
    std::ifstream in(path, std::ios::binary);
    return static_cast<bool>(in);
}

/// Minimal JSON number/bool/string extractors for our fixed sidecar schema.
static bool json_find_int(const std::string& s, const char* key, long long& out)
{
    const std::string pat = std::string("\"") + key + "\"";
    size_t p = s.find(pat);
    if (p == std::string::npos) return false;
    p = s.find(':', p + pat.size());
    if (p == std::string::npos) return false;
    ++p;
    while (p < s.size() && (s[p] == ' ' || s[p] == '\t')) ++p;
    try {
        size_t idx = 0;
        out = std::stoll(s.substr(p), &idx);
        return idx > 0;
    } catch (...) {
        return false;
    }
}

static bool json_find_bool(const std::string& s, const char* key, bool& out)
{
    const std::string pat = std::string("\"") + key + "\"";
    size_t p = s.find(pat);
    if (p == std::string::npos) return false;
    p = s.find(':', p + pat.size());
    if (p == std::string::npos) return false;
    ++p;
    while (p < s.size() && (s[p] == ' ' || s[p] == '\t')) ++p;
    if (s.compare(p, 4, "true") == 0) {
        out = true;
        return true;
    }
    if (s.compare(p, 5, "false") == 0) {
        out = false;
        return true;
    }
    return false;
}

static bool json_find_string(const std::string& s, const char* key, std::string& out)
{
    const std::string pat = std::string("\"") + key + "\"";
    size_t p = s.find(pat);
    if (p == std::string::npos) return false;
    p = s.find(':', p + pat.size());
    if (p == std::string::npos) return false;
    p = s.find('"', p + 1);
    if (p == std::string::npos) return false;
    const size_t q = s.find('"', p + 1);
    if (q == std::string::npos) return false;
    out = s.substr(p + 1, q - p - 1);
    return true;
}

void Readout::SaveHcnnModel(const std::string& path_stem) const
{
    assert(net_);
    if (path_stem.empty())
        throw std::invalid_argument("Readout::SaveHcnnModel: empty path_stem");

    const std::string stem = path_stem_normalized(path_stem);
    const std::string hcnw_path = stem + ".hcnw";
    const std::string arch_path = stem + ".arch.json";

    hcnn::save_weights(*net_, hcnw_path);

    const auto layers = make_layer_specs(config_);
    const auto sum = hcnn::summarize_arch(
        static_cast<int>(config_.dim), config_.num_outputs, /*input_channels=*/1,
        layers);

    std::ofstream out(arch_path);
    if (!out)
        throw std::runtime_error(
            "Readout::SaveHcnnModel: cannot open " + arch_path);

    out << "{\n"
        << "  \"format\": \"" << kArchSidecarFormat << "\",\n"
        << "  \"version\": " << kArchSidecarVersion << ",\n"
        << "  \"start_dim\": " << config_.dim << ",\n"
        << "  \"num_outputs\": " << config_.num_outputs << ",\n"
        << "  \"input_channels\": 1,\n"
        << "  \"task\": \""
        << (config_.task == ReadoutTask::Classification ? "classification"
                                                        : "regression")
        << "\",\n"
        << "  \"num_layers\": " << config_.num_layers << ",\n"
        << "  \"use_pooling\": " << (config_.use_pooling ? "true" : "false")
        << ",\n"
        << "  \"pool_type\": \"" << pool_token(config_.pool_type) << "\",\n"
        << "  \"conv_channels\": " << config_.conv_channels << ",\n"
        << "  \"channel_growth\": " << config_.channel_growth << ",\n"
        << "  \"use_batchnorm\": "
        << (config_.use_batchnorm ? "true" : "false") << ",\n"
        << "  \"optimizer\": \"" << optimizer_token(config_.optimizer)
        << "\",\n"
        << "  \"activation\": \"" << activation_token(config_.activation)
        << "\",\n"
        << "  \"weight_count\": " << net_->GetWeightCount() << ",\n"
        << "  \"param_total\": " << sum.total << ",\n"
        << "  \"layers\": [\n";

    for (size_t i = 0; i < layers.size(); ++i) {
        const auto& L = layers[i];
        out << "    {";
        if (L.kind == hcnn::LayerSpec::Kind::Conv) {
            out << "\"kind\": \"conv\", \"c_out\": " << L.c_out
                << ", \"activation\": \"" << hcnn_act_token(L.activation)
                << "\", \"bias\": " << (L.use_bias ? "true" : "false")
                << ", \"bn\": " << (L.use_bn ? "true" : "false");
        } else {
            out << "\"kind\": \"pool\", \"type\": \""
                << hcnn_pool_token(L.pool_type) << "\"";
        }
        out << "}" << (i + 1 < layers.size() ? "," : "") << "\n";
    }
    out << "  ]\n"
        << "}\n";
    if (!out)
        throw std::runtime_error(
            "Readout::SaveHcnnModel: write failed for " + arch_path);
}

void Readout::LoadHcnnModel(const std::string& path_stem, ReadoutLoadMode mode)
{
    assert(net_);
    if (path_stem.empty())
        throw std::invalid_argument("Readout::LoadHcnnModel: empty path_stem");

    const std::string stem = path_stem_normalized(path_stem);
    const std::string hcnw_path = stem + ".hcnw";
    const std::string arch_path = stem + ".arch.json";

    if (!file_exists(hcnw_path))
        throw std::runtime_error(
            "Readout::LoadHcnnModel: missing " + hcnw_path);

    // Validate arch sidecar against the live readout when present.
    if (file_exists(arch_path)) {
        std::ifstream in(arch_path);
        if (!in)
            throw std::runtime_error(
                "Readout::LoadHcnnModel: cannot open " + arch_path);
        std::ostringstream ss;
        ss << in.rdbuf();
        const std::string text = ss.str();

        std::string format;
        if (!json_find_string(text, "format", format)
            || format != kArchSidecarFormat) {
            throw std::runtime_error(
                "Readout::LoadHcnnModel: " + arch_path
                + " is not a " + std::string(kArchSidecarFormat)
                + " sidecar");
        }

        long long version = 0;
        if (!json_find_int(text, "version", version)
            || version != kArchSidecarVersion) {
            throw std::runtime_error(
                "Readout::LoadHcnnModel: unsupported arch sidecar version in "
                + arch_path + " (need " + std::to_string(kArchSidecarVersion)
                + ")");
        }

        auto require_int = [&](const char* key, long long expect) {
            long long v = 0;
            if (!json_find_int(text, key, v) || v != expect) {
                throw std::runtime_error(
                    std::string("Readout::LoadHcnnModel: arch mismatch on '")
                    + key + "' (file=" + std::to_string(v)
                    + ", live=" + std::to_string(expect) + ")");
            }
        };
        auto require_bool = [&](const char* key, bool expect) {
            bool v = false;
            if (!json_find_bool(text, key, v) || v != expect) {
                throw std::runtime_error(
                    std::string("Readout::LoadHcnnModel: arch mismatch on '")
                    + key + "'");
            }
        };
        auto require_str = [&](const char* key, const std::string& expect) {
            std::string v;
            if (!json_find_string(text, key, v) || v != expect) {
                throw std::runtime_error(
                    std::string("Readout::LoadHcnnModel: arch mismatch on '")
                    + key + "' (file='" + v + "', live='" + expect + "')");
            }
        };

        require_int("start_dim", static_cast<long long>(config_.dim));
        require_int("num_outputs", config_.num_outputs);
        require_int("input_channels", 1);
        require_str("task",
                    config_.task == ReadoutTask::Classification
                        ? "classification"
                        : "regression");
        // num_layers 0 means auto in config — compare resolved layer count via
        // weight_count / expanded stack rather than raw knob when 0.
        if (config_.num_layers > 0)
            require_int("num_layers", config_.num_layers);
        require_bool("use_pooling", config_.use_pooling);
        require_str("pool_type", pool_token(config_.pool_type));
        require_int("conv_channels", config_.conv_channels);
        require_int("channel_growth", config_.channel_growth);
        require_bool("use_batchnorm", config_.use_batchnorm);
        require_str("activation", activation_token(config_.activation));

        long long file_wc = 0;
        if (json_find_int(text, "weight_count", file_wc)
            && static_cast<size_t>(file_wc) != net_->GetWeightCount()) {
            throw std::runtime_error(
                "Readout::LoadHcnnModel: weight_count mismatch (file="
                + std::to_string(file_wc) + ", live="
                + std::to_string(net_->GetWeightCount()) + ")");
        }
    }

    const bool reset_moments = (mode == ReadoutLoadMode::ResumeTrain);
    hcnn::load_weights(*net_, hcnw_path, reset_moments);
    trained_ = true;
}

std::string Readout::ArchSummary() const
{
    assert(net_);
    const auto layers = make_layer_specs(config_);
    const auto sum = hcnn::summarize_arch(
        static_cast<int>(config_.dim), config_.num_outputs, /*input_channels=*/1,
        layers);
    std::ostringstream os;
    hcnn::print_arch(os, static_cast<int>(config_.dim), config_.num_outputs,
                     /*input_channels=*/1, layers, sum);
    os << "HCNN live weight_count: " << net_->GetWeightCount() << "\n";
    return os.str();
}
