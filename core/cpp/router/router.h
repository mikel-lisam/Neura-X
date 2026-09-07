// ==========================================================
// Neura-X: Intelligence Without Limits.
// Copyright (c) 2026 Edusei Mikel Lisamba. All Rights Reserved.
//
// Liquid Router — Header
// ==========================================================

#ifndef NEURA_X_ROUTER_H
#define NEURA_X_ROUTER_H

#include <cstdint>
#include <cstddef>
#include <vector>
#include <string>
#include <memory>

namespace neurax {
namespace router {

// ──────────────────────────────────────────────────────────
// Gate State
// ──────────────────────────────────────────────────────────

enum class GateState : uint8_t {
    ASLEEP = 0,
    AWAKE = 1,
    TRANSITIONING = 2
};

// ──────────────────────────────────────────────────────────
// Routing Statistics
// ──────────────────────────────────────────────────────────

struct RoutingStats {
    uint64_t total_routed;
    uint64_t total_awake;
    uint64_t total_asleep;
    double sparsity_ratio;
    double avg_awake_fraction;
};

// ──────────────────────────────────────────────────────────
// Liquid Router
// ──────────────────────────────────────────────────────────

class LiquidRouter {
public:
    LiquidRouter(
        int num_neurons,
        float awake_threshold = 0.05f,
        float temperature = 1.0f,
        float temperature_decay = 0.999f
    );
    ~LiquidRouter();

    // Route an input and produce a gate mask.
    // Returns the number of awake neurons.
    int route(const float* input, int input_size, uint8_t* gate_mask);

    // Step the router (decay temperature after each training step).
    void step();

    // Set the awake threshold.
    void set_awake_threshold(float threshold);

    // Get current temperature.
    float temperature() const { return temperature_; }

    // Get routing statistics.
    RoutingStats get_stats() const;

    // Reset statistics.
    void reset_stats();

    // Get the number of neurons.
    int num_neurons() const { return num_neurons_; }

    // Get the expected number of awake neurons.
    int expected_awake() const;

private:
    int num_neurons_;
    float awake_threshold_;
    float temperature_;
    float temperature_decay_;
    int step_count_;

    // Gate logits (learned probabilities)
    std::vector<float> gate_logits_;

    // Statistics
    uint64_t total_routed_;
    uint64_t total_awake_;

    // Internal: Apply Gumbel-Softmax sampling.
    void gumbel_softmax(const float* logits, float temp,
                        float* gates, int n, bool hard);
};

// ──────────────────────────────────────────────────────────
// Sparsity Engine (Manages memory paging based on gates)
// ──────────────────────────────────────────────────────────

class SparsityEngine {
public:
    SparsityEngine(LiquidRouter* router);
    ~SparsityEngine();

    // Apply sparsity: page out sleeping neurons, page in awake ones.
    void apply(const uint8_t* gate_mask, int mask_size);

    // Get memory savings.
    struct MemorySavings {
        size_t total_params;
        size_t awake_params;
        size_t asleep_params;
        double reduction_factor;
        size_t ram_saved_bytes;
    };

    MemorySavings get_memory_savings(int param_size_bytes) const;

private:
    LiquidRouter* router_;
    std::vector<GateState> neuron_states_;
};

} // namespace router
} // namespace neurax

#endif // NEURA_X_ROUTER_H