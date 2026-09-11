// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 David Liptak

#include "HCNNReadout.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace hcnn {

HCNNReadout::HCNNReadout(int nc, int nf)
    : num_outputs(nc), num_features(nf),
      weights(static_cast<size_t>(nc) * static_cast<size_t>(nf), 0.0f),
      bias(nc, 0.0f),
      weight_m(static_cast<size_t>(nc) * static_cast<size_t>(nf), 0.0f),
      bias_m(nc, 0.0f) {
    if (num_outputs < 1) {
        throw std::runtime_error("HCNNReadout requires num_outputs >= 1");
    }
    if (num_features < 1) {
        throw std::runtime_error("HCNNReadout requires num_features >= 1");
    }
}

void HCNNReadout::randomize_weights(float scale, std::mt19937& rng) {
    // Xavier/Glorot uniform for the linear layer.
    if (scale <= 0.0f) {
        scale = std::sqrt(6.0f / static_cast<float>(num_features + num_outputs));
    }
    std::uniform_real_distribution<float> dist(-scale, scale);
    for (auto& w : weights) {
        w = dist(rng);
    }
    for (auto& b : bias) b = 0.0f;
    std::fill(weight_m.begin(), weight_m.end(), 0.0f);
    std::fill(bias_m.begin(), bias_m.end(), 0.0f);
    std::fill(weight_m2.begin(), weight_m2.end(), 0.0f);
    std::fill(bias_m2.begin(), bias_m2.end(), 0.0f);
}

void HCNNReadout::set_optimizer(OptimizerType type, float beta1, float beta2, float eps) {
    optimizer_type_ = type;
    adam_beta1_ = beta1;
    adam_beta2_ = beta2;
    adam_eps_ = eps;
    if (type == OptimizerType::ADAM) {
        weight_m2.assign(weights.size(), 0.0f);
        bias_m2.assign(bias.size(), 0.0f);
    } else {
        weight_m2.clear(); weight_m2.shrink_to_fit();
        bias_m2.clear(); bias_m2.shrink_to_fit();
    }
}

void HCNNReadout::clear_optimizer_moments() {
    std::fill(weight_m.begin(), weight_m.end(), 0.0f);
    std::fill(bias_m.begin(), bias_m.end(), 0.0f);
    std::fill(weight_m2.begin(), weight_m2.end(), 0.0f);
    std::fill(bias_m2.begin(), bias_m2.end(), 0.0f);
}

void HCNNReadout::forward(const float* in, float* out) const {
    for (int o = 0; o < num_outputs; ++o) {
        float sum = bias[o];
        const float* wrow = weights.data() + static_cast<size_t>(o) * num_features;
        for (int f = 0; f < num_features; ++f) {
            sum += wrow[f] * in[f];
        }
        out[o] = sum;
    }
}

void HCNNReadout::fill_grad_in(const float* grad_logits, float* grad_in) const {
    if (grad_in_loop_ == ReadoutGradInLoop::FeatureOuter) {
        // Legacy A/B: for each feature, sum over outputs (column of W).
        for (int f = 0; f < num_features; ++f) {
            float g = 0.0f;
            for (int o = 0; o < num_outputs; ++o) {
                g += grad_logits[o] * weights[static_cast<size_t>(o) * num_features + f];
            }
            grad_in[f] = g;
        }
        return;
    }

    // OutputOuter (default): stream each weight row into grad_in.
    std::fill(grad_in, grad_in + num_features, 0.0f);
    for (int o = 0; o < num_outputs; ++o) {
        const float go = grad_logits[o];
        const float* wrow = weights.data() + static_cast<size_t>(o) * num_features;
        for (int f = 0; f < num_features; ++f) {
            grad_in[f] += go * wrow[f];
        }
    }
}

void HCNNReadout::backward(const float* grad_logits, const float* in,
                           float* grad_in, float learning_rate, float momentum,
                           float weight_decay, int timestep) {
    const bool use_adam = (optimizer_type_ == OptimizerType::ADAM && timestep > 0);
    const float bc1 = use_adam ? 1.0f - static_cast<float>(std::pow(adam_beta1_, timestep)) : 1.0f;
    const float bc2 = use_adam ? 1.0f - static_cast<float>(std::pow(adam_beta2_, timestep)) : 1.0f;

    if (grad_in) {
        fill_grad_in(grad_logits, grad_in);
    }

    for (int o = 0; o < num_outputs; ++o) {
        const float go = grad_logits[o];
        float* wrow = weights.data() + static_cast<size_t>(o) * num_features;
        float* mrow = weight_m.data() + static_cast<size_t>(o) * num_features;
        for (int f = 0; f < num_features; ++f) {
            float g = go * in[f];
            if (use_adam) {
                float* m2row = weight_m2.data() + static_cast<size_t>(o) * num_features;
                mrow[f] = adam_beta1_ * mrow[f] + (1.0f - adam_beta1_) * g;
                m2row[f] = adam_beta2_ * m2row[f] + (1.0f - adam_beta2_) * g * g;
                float mh = mrow[f] / bc1;
                float vh = m2row[f] / bc2;
                wrow[f] -= learning_rate * (mh / (std::sqrt(vh) + adam_eps_) + weight_decay * wrow[f]);
            } else {
                g += weight_decay * wrow[f];
                mrow[f] = momentum * mrow[f] + g;
                wrow[f] -= learning_rate * mrow[f];
            }
        }
        if (use_adam) {
            bias_m[o] = adam_beta1_ * bias_m[o] + (1.0f - adam_beta1_) * go;
            bias_m2[o] = adam_beta2_ * bias_m2[o] + (1.0f - adam_beta2_) * go * go;
            float mh = bias_m[o] / bc1;
            float vh = bias_m2[o] / bc2;
            bias[o] -= learning_rate * mh / (std::sqrt(vh) + adam_eps_);
        } else {
            bias_m[o] = momentum * bias_m[o] + go;
            bias[o] -= learning_rate * bias_m[o];
        }
    }
}

void HCNNReadout::compute_gradients(const float* grad_logits, const float* in,
                                    float* grad_in, float* weight_grad,
                                    float* bias_grad) const {
    if (grad_in) {
        fill_grad_in(grad_logits, grad_in);
    }

    // weight_grad / bias_grad may be null (e.g. microbench of grad_in only).
    if (!weight_grad && !bias_grad) {
        return;
    }

    for (int o = 0; o < num_outputs; ++o) {
        const float go = grad_logits[o];
        if (weight_grad) {
            float* wg = weight_grad + static_cast<size_t>(o) * num_features;
            for (int f = 0; f < num_features; ++f) {
                wg[f] = go * in[f];
            }
        }
        if (bias_grad) bias_grad[o] = go;
    }
}

void HCNNReadout::apply_gradients(const float* weight_grad, const float* bias_grad,
                                  float learning_rate, float momentum, float weight_decay,
                                  int timestep) {
    const bool use_adam = (optimizer_type_ == OptimizerType::ADAM && timestep > 0);
    const float bc1 = use_adam ? 1.0f - static_cast<float>(std::pow(adam_beta1_, timestep)) : 1.0f;
    const float bc2 = use_adam ? 1.0f - static_cast<float>(std::pow(adam_beta2_, timestep)) : 1.0f;
    const int total_w = num_outputs * num_features;

    if (use_adam) {
        for (int i = 0; i < total_w; ++i) {
            float g = weight_grad[i];
            weight_m[i] = adam_beta1_ * weight_m[i] + (1.0f - adam_beta1_) * g;
            weight_m2[i] = adam_beta2_ * weight_m2[i] + (1.0f - adam_beta2_) * g * g;
            float mh = weight_m[i] / bc1;
            float vh = weight_m2[i] / bc2;
            weights[i] -= learning_rate * (mh / (std::sqrt(vh) + adam_eps_) + weight_decay * weights[i]);
        }
    } else {
        for (int i = 0; i < total_w; ++i) {
            float g = weight_grad[i] + weight_decay * weights[i];
            weight_m[i] = momentum * weight_m[i] + g;
            weights[i] -= learning_rate * weight_m[i];
        }
    }

    if (bias_grad) {
        for (int o = 0; o < num_outputs; ++o) {
            if (use_adam) {
                float g = bias_grad[o];
                bias_m[o] = adam_beta1_ * bias_m[o] + (1.0f - adam_beta1_) * g;
                bias_m2[o] = adam_beta2_ * bias_m2[o] + (1.0f - adam_beta2_) * g * g;
                float mh = bias_m[o] / bc1;
                float vh = bias_m2[o] / bc2;
                bias[o] -= learning_rate * mh / (std::sqrt(vh) + adam_eps_);
            } else {
                bias_m[o] = momentum * bias_m[o] + bias_grad[o];
                bias[o] -= learning_rate * bias_m[o];
            }
        }
    }
}

} // namespace hcnn
