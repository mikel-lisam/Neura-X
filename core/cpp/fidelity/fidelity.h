// ==========================================================
// Neura-X: Intelligence Without Limits.
// Copyright (c) 2026 Edusei Mikel Lisamba. All Rights Reserved.
//
// Dynamic Fidelity Engine — Header
// ==========================================================

#ifndef NEURA_X_FIDELITY_H
#define NEURA_X_FIDELITY_H

#include <cstdint>
#include <cstddef>
#include <vector>
#include <string>

namespace neurax {
namespace fidelity {

// ──────────────────────────────────────────────────────────
// Fidelity Report
// ──────────────────────────────────────────────────────────

struct FidelityReport {
    double fractal_error;       // Fractal approximation error
    double shadow_error;        // Shadow optimizer projection error
    double ghost_error;         // Ghost tensor reconstruction error
    double predictive_error;    // Predictive coding error
    double total_error;         // Combined error
    bool within_bounds;         // Whether all errors are within bounds
};

// ──────────────────────────────────────────────────────────
// Error Bound Tracker
// ──────────────────────────────────────────────────────────

class ErrorBoundTracker {
public:
    ErrorBoundTracker();
    ~ErrorBoundTracker();

    // Set error bounds for each paradigm.
    void set_bounds(
        double fractal_bound,
        double shadow_bound,
        double ghost_bound,
        double predictive_bound
    );

    // Record an error measurement.
    void record(
        double fractal_error,
        double shadow_error,
        double ghost_error,
        double predictive_error
    );

    // Get the current fidelity report.
    FidelityReport get_report() const;

    // Check if all errors are within bounds.
    bool is_within_bounds() const;

    // Get total number of measurements.
    uint64_t measurement_count() const { return measurement_count_; }

private:
    double fractal_bound_;
    double shadow_bound_;
    double ghost_bound_;
    double predictive_bound_;

    // Running averages
    double avg_fractal_error_;
    double avg_shadow_error_;
    double avg_ghost_error_;
    double avg_predictive_error_;
    uint64_t measurement_count_;
};

// ──────────────────────────────────────────────────────────
// Dynamic Fidelity Engine
// ──────────────────────────────────────────────────────────

class DynamicFidelityEngine {
public:
    DynamicFidelityEngine();
    ~DynamicFidelityEngine();

    // Initialize with default bounds.
    void initialize();

    // Evaluate fidelity and adjust parameters if needed.
    // Returns the fidelity report.
    FidelityReport evaluate(
        const float* weights, int weight_count,
        const float* gradients, int grad_count,
        int current_rank
    );

    // Get recommended rank adjustment for Shadow Optimizer.
    int recommend_rank(int current_rank, double gradient_energy) const;

    // Get recommended K adjustment for Fractal Tensors.
    int recommend_K(int current_K, double approximation_error) const;

    // Get the error bound tracker.
    const ErrorBoundTracker& tracker() const { return tracker_; }

private:
    ErrorBoundTracker tracker_;
    bool initialized_;
};

} // namespace fidelity
} // namespace neurax

#endif // NEURA_X_FIDELITY_H