// ==========================================================
// Neura-X: Intelligence Without Limits.
// Copyright (c) 2026 Edusei Mikel Lisamba. All Rights Reserved.
//
// Error Bound Tracker — Implementation
// ==========================================================

#include "fidelity.h"
#include <cmath>

namespace neurax {
namespace fidelity {

ErrorBoundTracker::ErrorBoundTracker()
    : fractal_bound_(0.001)
    , shadow_bound_(0.01)
    , ghost_bound_(0.01)
    , predictive_bound_(0.01)
    , avg_fractal_error_(0.0)
    , avg_shadow_error_(0.0)
    , avg_ghost_error_(0.0)
    , avg_predictive_error_(0.0)
    , measurement_count_(0)
{
}

ErrorBoundTracker::~ErrorBoundTracker()
{
}

void ErrorBoundTracker::set_bounds(
    double fractal_bound,
    double shadow_bound,
    double ghost_bound,
    double predictive_bound)
{
    fractal_bound_ = fractal_bound;
    shadow_bound_ = shadow_bound;
    ghost_bound_ = ghost_bound;
    predictive_bound_ = predictive_bound;
}

void ErrorBoundTracker::record(
    double fractal_error,
    double shadow_error,
    double ghost_error,
    double predictive_error)
{
    measurement_count_++;

    // Running average
    double n = static_cast<double>(measurement_count_);
    avg_fractal_error_ += (fractal_error - avg_fractal_error_) / n;
    avg_shadow_error_ += (shadow_error - avg_shadow_error_) / n;
    avg_ghost_error_ += (ghost_error - avg_ghost_error_) / n;
    avg_predictive_error_ += (predictive_error - avg_predictive_error_) / n;
}

FidelityReport ErrorBoundTracker::get_report() const
{
    FidelityReport report;
    report.fractal_error = avg_fractal_error_;
    report.shadow_error = avg_shadow_error_;
    report.ghost_error = avg_ghost_error_;
    report.predictive_error = avg_predictive_error_;
    report.total_error = avg_fractal_error_ + avg_shadow_error_
                       + avg_ghost_error_ + avg_predictive_error_;
    report.within_bounds = is_within_bounds();
    return report;
}

bool ErrorBoundTracker::is_within_bounds() const
{
    return (avg_fractal_error_ <= fractal_bound_)
        && (avg_shadow_error_ <= shadow_bound_)
        && (avg_ghost_error_ <= ghost_bound_)
        && (avg_predictive_error_ <= predictive_bound_);
}

} // namespace fidelity
} // namespace neurax