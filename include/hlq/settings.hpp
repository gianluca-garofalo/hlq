#pragma once

/// @file
/// Tolerances and limits of the solver.

#include <type_traits>

namespace hlq {

/// Default tolerance of a scalar type: 1e-9 for double, 1e-5 for float.
template <typename Scalar>
constexpr Scalar default_tolerance() noexcept {
    if constexpr (std::is_same_v<Scalar, float>) {
        return Scalar(1e-5);
    } else {
        return Scalar(1e-9);
    }
}

/// Tolerances and limits of `Solver`. The defaults suit double precision data of order one.
template <typename Scalar = double>
struct Settings {
    /// A bound is violated when it is exceeded by more than this (absolute, in the units of the rows).
    Scalar feasibility_tolerance = default_tolerance<Scalar>();
    /// A row is dependent on the active rows before it when what it has outside them is shorter than this
    /// times the longest row of its level.
    Scalar rank_tolerance = default_tolerance<Scalar>();
    /// A multiplier is nonzero when its magnitude is above this times one plus the largest multiplier of its
    /// subproblem.
    Scalar dual_tolerance = default_tolerance<Scalar>();
    /// Maximum number of active-set changes (activations plus deactivations) in one solve.
    int max_changes = 500;
    /// Take a last level of equalities with full column rank, such as x = 0, as the choice among the optima
    /// of the levels above instead of searching it: the same solution, at no cost.
    bool fold_final_level = true;
    /// Gram mode: keep only the triangular factor of the active rows, as DAQP does, not an orthonormal basis
    /// of them. Each active-set change costs less, but the factor comes from inner products of the rows,
    /// which square their conditioning: ranks are decided only to about sqrt(epsilon) (the rank tolerance is
    /// raised to that), and ill-conditioned levels lose accuracy.
    bool gram_mode = false;
};

}  // namespace hlq
