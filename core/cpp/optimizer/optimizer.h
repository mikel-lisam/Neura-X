// ==========================================================
// Neura-X: Intelligence Without Limits.
// Copyright (c) 2026 Edusei Mikel Lisamba. All Rights Reserved.
//
// Shadow Optimizer — Header
// ==========================================================

#ifndef NEURA_X_OPTIMIZER_H
#define NEURA_X_OPTIMIZER_H

#include <cstdint>
#include <cstddef>
#include <vector>
#include <string>
#include <memory>

namespace neurax {
namespace optimizer {

// ──────────────────────────────────────────────────────────
// Shadow Optimizer State
// ──────────────────────────────────────────────────────────

struct ShadowState {
    std::vector<float> m_sub;   // Shadow momentum (r × r)
    std::vector<float> v_sub;   // Shadow variance (r × r)
    std::vector<float> P;       // Row projection (m × r)
    std::vector<float> Q;       // Column projection (n × r)
    int m, n, r;               // Dimensions
    int step_count;

    ShadowState() : m(0), n(0), r(0), step_count(0) {}
};

// ──────────────────────────────────────────────────────────
// Shadow AdamW Optimizer
// ──────────────────────────────────────────────────────────

class ShadowAdamW {
public:
    ShadowAdamW(
        int param_rows, int param_cols,
        int rank = 64,
        float lr = 1e-3f,
        float beta1 = 0.9f,
        float beta2 = 0.999f,
        float eps = 1e-8f,
        float weight_decay = 0.01f,
        int rotation_interval = 100
    );
    ~ShadowAdamW();

    // Perform one optimization step.
    // grad: the full gradient (param_rows × param_cols)
    // param: the parameter matrix to update (modified in-place)
    void step(const float* grad, float* param);

    // Perform Dynamic Subspace Rotation.
    void rotate_subspace();

    // Get memory usage comparison.
    struct MemoryUsage {
        size_t standard_bytes;
        size_t shadow_bytes;
        double reduction_factor;
    };
    MemoryUsage get_memory_usage() const;

    // Get current step count.
    int step_count() const { return state_.step_count; }

private:
    ShadowState state_;
    float lr_;
    float beta1_, beta2_;
    float eps_;
    float weight_decay_;
    int rotation_interval_;

    // Internal: Project gradient into subspace.
    void project_gradient(const float* grad, std::vector<float>& grad_sub);

    // Internal: Project update back to full space.
    void project_update(const std::vector<float>& update_sub, float* update);
};

// ──────────────────────────────────────────────────────────
// Subspace Rotation Engine
// ──────────────────────────────────────────────────────────

class SubspaceRotation {
public:
    SubspaceRotation(int rank);
    ~SubspaceRotation();

    // Perform rotation using accumulated gradients.
    void rotate(ShadowState& state, const std::vector<float>& accumulated_grad);

    // Get rotation count.
    int rotation_count() const { return rotation_count_; }

private:
    int rank_;
    int rotation_count_;
};

} // namespace optimizer
} // namespace neurax

#endif // NEURA_X_OPTIMIZER_H