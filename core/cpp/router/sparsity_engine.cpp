// ==========================================================
// Neura-X: Intelligence Without Limits.
// Copyright (c) 2026 Edusei Mikel Lisamba. All Rights Reserved.
//
// Sparsity Engine — Manages memory based on router gates
// ==========================================================

#include "router.h"
#include <cstring>

namespace neurax {
namespace router {

SparsityEngine::SparsityEngine(LiquidRouter* router)
    : router_(router)
{
    if (router_) {
        neuron_states_.resize(router_->num_neurons(), GateState::ASLEEP);
    }
}

SparsityEngine::~SparsityEngine()
{
}

void SparsityEngine::apply(const uint8_t* gate_mask, int mask_size)
{
    if (!gate_mask || !router_ || mask_size <= 0) return;

    int n = std::min(mask_size, static_cast<int>(neuron_states_.size()));

    for (int i = 0; i < n; i++) {
        if (gate_mask[i]) {
            if (neuron_states_[i] == GateState::ASLEEP) {
                neuron_states_[i] = GateState::TRANSITIONING;
                // In production: trigger async page-in from disk
                neuron_states_[i] = GateState::AWAKE;
            }
        } else {
            if (neuron_states_[i] == GateState::AWAKE) {
                neuron_states_[i] = GateState::TRANSITIONING;
                // In production: trigger async page-out to disk
                neuron_states_[i] = GateState::ASLEEP;
            }
        }
    }
}

SparsityEngine::MemorySavings SparsityEngine::get_memory_savings(
    int param_size_bytes) const
{
    MemorySavings savings;
    savings.total_params = neuron_states_.size();
    savings.awake_params = 0;
    savings.asleep_params = 0;

    for (size_t i = 0; i < neuron_states_.size(); i++) {
        if (neuron_states_[i] == GateState::AWAKE) {
            savings.awake_params++;
        } else {
            savings.asleep_params++;
        }
    }

    savings.reduction_factor = (savings.awake_params > 0)
        ? static_cast<double>(savings.total_params) / savings.awake_params
        : 1.0;

    savings.ram_saved_bytes = savings.asleep_params * param_size_bytes;

    return savings;
}

} // namespace router
} // namespace neurax