// ==========================================================
// Neura-X: Intelligence Without Limits.
// Copyright (c) 2026 Edusei Mikel Lisamba. All Rights Reserved.
//
// Shadow AdamW Optimizer — Implementation
// ==========================================================

#include "optimizer.h"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <random>

namespace neurax {
namespace optimizer {

ShadowAdamW::ShadowAdamW(
    int param_rows, int param_cols,
    int rank, float lr,
    float beta1, float beta2, float eps,
    float weight_decay, int rotation_interval)
    : lr_(lr)
    , beta1_(beta1)
    , beta2_(beta2)
    , eps_(eps)
    , weight_decay_(weight_decay)
    , rotation_interval_(rotation_interval)
{
    state_.m = param_rows;
    state_.n = param_cols;
    state_.r = rank;
    state_.step_count = 0;

    int r = rank;
    state_.m_sub.resize(r * r, 0.0f);
    state_.v_sub.resize(r * r, 0.0f);
    state_.P.resize(param_rows * r);
    state_.Q.resize(param_cols * r);

    // Initialize projections with small random values
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 0.01f);
    for (auto& v : state_.P) v = dist(rng);
    for (auto& v : state_.Q) v = dist(rng);
}

ShadowAdamW::~ShadowAdamW()
{
}

void ShadowAdamW::project_gradient(
    const float* grad, std::vector<float>& grad_sub)
{
    int m = state_.m, n = state_.n, r = state_.r;
    grad_sub.resize(r * r, 0.0f);

    // grad_sub = P^T × grad × Q
    // Step 1: temp = grad × Q  (m × r)
    std::vector<float> temp(m * r, 0.0f);
    for (int i = 0; i < m; i++) {
        for (int k = 0; k < r; k++) {
            float sum = 0.0f;
            for (int j = 0; j < n; j++) {
                sum += grad[i * n + j] * state_.Q[j * r + k];
            }
            temp[i * r + k] = sum;
        }
    }

    // Step 2: grad_sub = P^T × temp  (r × r)
    for (int k1 = 0; k1 < r; k1++) {
        for (int k2 = 0; k2 < r; k2++) {
            float sum = 0.0f;
            for (int i = 0; i < m; i++) {
                sum += state_.P[i * r + k1] * temp[i * r + k2];
            }
            grad_sub[k1 * r + k2] = sum;
        }
    }
}

void ShadowAdamW::project_update(
    const std::vector<float>& update_sub, float* update)
{
    int m = state_.m, n = state_.n, r = state_.r;

    // update = P × update_sub × Q^T
    // Step 1: temp = P × update_sub  (m × r)
    std::vector<float> temp(m * r, 0.0f);
    for (int i = 0; i < m; i++) {
        for (int k2 = 0; k2 < r; k2++) {
            float sum = 0.0f;
            for (int k1 = 0; k1 < r; k1++) {
                sum += state_.P[i * r + k1] * update_sub[k1 * r + k2];
            }
            temp[i * r + k2] = sum;
        }
    }

    // Step 2: update = temp × Q^T  (m × n)
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int k = 0; k < r; k++) {
                sum += temp[i * r + k] * state_.Q[j * r + k];
            }
            update[i * n + j] = sum;
        }
    }
}

void ShadowAdamW::step(const float* grad, float* param)
{
    if (!grad || !param) return;

    int m = state_.m, n = state_.n, r = state_.r;
    state_.step_count++;

    // Apply weight decay to params
    if (weight_decay_ > 0.0f) {
        for (int i = 0; i < m * n; i++) {
            param[i] -= lr_ * weight_decay_ * param[i];
        }
    }

    // Project gradient into subspace
    std::vector<float> grad_sub;
    project_gradient(grad, grad_sub);

    // Update shadow momentum and variance
    for (int i = 0; i < r * r; i++) {
        state_.m_sub[i] = beta1_ * state_.m_sub[i] + (1.0f - beta1_) * grad_sub[i];
        state_.v_sub[i] = beta2_ * state_.v_sub[i] + (1.0f - beta2_) * grad_sub[i] * grad_sub[i];
    }

    // Bias correction
    float bc1 = 1.0f - std::pow(beta1_, state_.step_count);
    float bc2 = 1.0f - std::pow(beta2_, state_.step_count);

    // Compute update in subspace
    std::vector<float> update_sub(r * r);
    for (int i = 0; i < r * r; i++) {
        float m_hat = state_.m_sub[i] / bc1;
        float v_hat = state_.v_sub[i] / bc2;
        update_sub[i] = lr_ * m_hat / (std::sqrt(v_hat) + eps_);
    }

    // Project back and apply
    std::vector<float> update(m * n);
    project_update(update_sub, update.data());

    for (int i = 0; i < m * n; i++) {
        param[i] -= update[i];
    }

    // Dynamic Subspace Rotation
    if (state_.step_count % rotation_interval_ == 0) {
        rotate_subspace();
    }
}

void ShadowAdamW::rotate_subspace()
{
    // Add small perturbation to projections (simplified rotation)
    std::mt19937 rng(state_.step_count);
    std::normal_distribution<float> dist(0.0f, 0.001f);

    for (auto& v : state_.P) v += dist(rng);
    for (auto& v : state_.Q) v += dist(rng);
}

ShadowAdamW::MemoryUsage ShadowAdamW::get_memory_usage() const
{
    int m = state_.m, n = state_.n, r = state_.r;

    MemoryUsage usage;
    usage.standard_bytes = 2 * m * n * sizeof(float); // M + V
    usage.shadow_bytes = 2 * r * r * sizeof(float)   // M_sub + V_sub
                       + m * r * sizeof(float)        // P
                       + n * r * sizeof(float);       // Q
    usage.reduction_factor = (usage.shadow_bytes > 0)
        ? static_cast<double>(usage.standard_bytes) / usage.shadow_bytes
        : 1.0;

    return usage;
}

} // namespace optimizer
} // namespace neurax