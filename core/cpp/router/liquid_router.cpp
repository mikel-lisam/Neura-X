// ==========================================================
// Neura-X: Intelligence Without Limits.
// Copyright (c) 2026 Edusei Mikel Lisamba. All Rights Reserved.
//
// Liquid Router — Implementation
// ==========================================================

#include "router.h"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <numeric>
#include <random>

namespace neurax {
namespace router {

LiquidRouter::LiquidRouter(
    int num_neurons,
    float awake_threshold,
    float temperature,
    float temperature_decay)
    : num_neurons_(num_neurons)
    , awake_threshold_(awake_threshold)
    , temperature_(temperature)
    , temperature_decay_(temperature_decay)
    , step_count_(0)
    , total_routed_(0)
    , total_awake_(0)
{
    gate_logits_.resize(num_neurons_, 0.0f);
}

LiquidRouter::~LiquidRouter()
{
}

void LiquidRouter::gumbel_softmax(
    const float* logits, float temp,
    float* gates, int n, bool hard)
{
    if (hard || temp < 0.01f) {
        for (int i = 0; i < n; i++) {
            gates[i] = (logits[i] > 0.0f) ? 1.0f : 0.0f;
        }
        return;
    }

    static std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0005f, 0.9995f);

    std::vector<float> noisy(n);
    for (int i = 0; i < n; i++) {
        float u = dist(rng);
        float gumbel = -std::log(-std::log(u));
        noisy[i] = (logits[i] + gumbel) / temp;
    }

    // Softmax
    float max_val = *std::max_element(noisy.begin(), noisy.end());
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        gates[i] = std::exp(noisy[i] - max_val);
        sum += gates[i];
    }
    if (sum > 1e-10f) {
        for (int i = 0; i < n; i++) {
            gates[i] /= sum;
        }
    }
}

int LiquidRouter::route(
    const float* input, int input_size, uint8_t* gate_mask)
{
    if (!gate_mask || num_neurons_ <= 0) return 0;
    (void)input;
    (void)input_size;

    // Compute soft gates
    std::vector<float> soft_gates(num_neurons_);
    bool hard = (temperature_ < 0.01f);
    gumbel_softmax(gate_logits_.data(), temperature_,
                   soft_gates.data(), num_neurons_, hard);

    // Apply sparsity: keep only top awake_threshold fraction
    int num_awake = std::max(1,
        static_cast<int>(num_neurons_ * awake_threshold_));

    // Find indices of top-K gates
    std::vector<int> indices(num_neurons_);
    std::iota(indices.begin(), indices.end(), 0);
    std::partial_sort(
        indices.begin(),
        indices.begin() + num_awake,
        indices.end(),
        [&soft_gates](int a, int b) {
            return soft_gates[a] > soft_gates[b];
        }
    );

    // Build gate mask
    std::memset(gate_mask, 0, num_neurons_);
    for (int i = 0; i < num_awake; i++) {
        gate_mask[indices[i]] = 1;
    }

    // Update statistics
    total_routed_ += num_neurons_;
    total_awake_ += num_awake;

    return num_awake;
}

void LiquidRouter::step()
{
    step_count_++;
    temperature_ *= temperature_decay_;
    if (temperature_ < 0.01f) temperature_ = 0.01f;
}

void LiquidRouter::set_awake_threshold(float threshold)
{
    if (threshold > 0.0f && threshold <= 1.0f) {
        awake_threshold_ = threshold;
    }
}

RoutingStats LiquidRouter::get_stats() const
{
    RoutingStats stats;
    stats.total_routed = total_routed_;
    stats.total_awake = total_awake_;
    stats.total_asleep = total_routed_ - total_awake_;
    stats.sparsity_ratio = (total_routed_ > 0)
        ? 1.0 - static_cast<double>(total_awake_) / total_routed_
        : 0.0;
    stats.avg_awake_fraction = (total_routed_ > 0)
        ? static_cast<double>(total_awake_) / total_routed_
        : 0.0;
    return stats;
}

void LiquidRouter::reset_stats()
{
    total_routed_ = 0;
    total_awake_ = 0;
}

int LiquidRouter::expected_awake() const
{
    return std::max(1, static_cast<int>(num_neurons_ * awake_threshold_));
}

} // namespace router
} // namespace neurax