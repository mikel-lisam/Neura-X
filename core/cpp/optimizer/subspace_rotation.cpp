// ==========================================================
// Neura-X: Intelligence Without Limits.
// Copyright (c) 2026 Edusei Mikel Lisamba. All Rights Reserved.
//
// Subspace Rotation Engine — Implementation
// ==========================================================

#include "optimizer.h"
#include <cmath>
#include <random>
#include <algorithm>

namespace neurax {
namespace optimizer {

SubspaceRotation::SubspaceRotation(int rank)
    : rank_(rank)
    , rotation_count_(0)
{
}

SubspaceRotation::~SubspaceRotation() {}

void SubspaceRotation::rotate(
    ShadowState& state,
    const std::vector<float>& accumulated_grad)
{
    if (accumulated_grad.empty()) return;

    // Calculate gradient magnitude for noise scaling
    float grad_norm = 0.0f;
    for (float g : accumulated_grad) {
        grad_norm += g * g;
    }
    grad_norm = std::sqrt(grad_norm);

    if (grad_norm < 1e-10f) return;

    float noise_scale = 0.001f * grad_norm;
    std::mt19937 rng(static_cast<unsigned int>(rotation_count_ + 42));
    std::normal_distribution<float> dist(0.0f, noise_scale);

    // Apply rotation noise to projection matrices
    for (auto& v : state.P) v += dist(rng);
    for (auto& v : state.Q) v += dist(rng);

    rotation_count_++;
}

} // namespace optimizer
} // namespace neurax