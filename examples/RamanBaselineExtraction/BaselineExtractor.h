#pragma once

#include "Cascade.h"

#include <span>
#include <string>

struct RamanSplit;

constexpr size_t kDim = 11;
constexpr size_t kN = size_t{1} << kDim;

inline CascadeConfig MakeBaseConfig()
{
    CascadeConfig cfg;
    cfg.dim = kDim;
    cfg.collect_threads = 1;

    cfg.exciter.seed = 3458567978345987ull;
    cfg.exciter.subcube_dim = 5;
    cfg.exciter.input_scaling = 1.0f;
    cfg.exciter.weight_scaling = 0.15;

/*****************************************************/
    cfg.interstage_scale = 5.0;

    cfg.T = 60;
    cfg.ic_seed = 1;
    cfg.reservoir.seed = 13871537636959942979ull;
    cfg.reservoir.spectral_radius = 0.95;
    cfg.reservoir.history_depth = 8;
    cfg.reservoir.leak_rate = 1.0;
    cfg.reservoir.input_scaling = 0.05;
    cfg.reservoir.bias_scaling = 0.001;

    cfg.readout_scale = 0.84;

    /*****************************************************/

    cfg.readout.epochs = 60;
    cfg.readout.num_outputs = static_cast<int>(kN);
    cfg.readout.task = ReadoutTask::Regression;
    cfg.readout.activation = ReadoutActivation::NONE;
    cfg.readout.batch_size = 48;
    cfg.readout.conv_channels = 1;
    cfg.readout.channel_growth = 1;
    cfg.readout.num_layers = 1;
    cfg.readout.use_pooling = false;
    cfg.readout.lr_max = 0.003f;
    cfg.readout.lr_min_frac = 0.04;
    cfg.readout.restore_best_epoch = true;

    return cfg;
}

class BaselineExtractor
{
public:
    BaselineExtractor();
    explicit BaselineExtractor(const CascadeConfig& cfg);
    ~BaselineExtractor() = default;

    BaselineExtractor(const BaselineExtractor&) = delete;
    BaselineExtractor& operator=(const BaselineExtractor&) = delete;
    BaselineExtractor(BaselineExtractor&&) = delete;
    BaselineExtractor& operator=(BaselineExtractor&&) = delete;

    [[nodiscard]] size_t Dim() const { return cascade_.Dim(); }
    [[nodiscard]] size_t N() const { return cascade_.N(); }
    [[nodiscard]] size_t NumOutputs() const { return cascade_.NumOutputs(); }

    [[nodiscard]] const CascadeConfig& config() const { return cascade_.config(); }
    [[nodiscard]] Cascade& cascade() { return cascade_; }
    [[nodiscard]] const Cascade& cascade() const { return cascade_; }

    void Collect(const RamanSplit& split);
    void Train();
    void Predict(std::span<const float> spectrum, std::span<float> baseline);

    void SaveReadout(const std::string& path_stem) const;
    void LoadReadout(const std::string& path_stem);

private:
    Cascade cascade_;
};
