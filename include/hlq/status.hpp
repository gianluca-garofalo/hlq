#pragma once

/// @file
/// Outcome of setting up and solving a problem.

namespace hlq {

/// Outcome of `Solver::set_problem`, `Solver::set_metric`, `Solver::set_reference` and `Solver::solve`.
enum class Status {
    /// The lexicographic optimum was found.
    Solved,
    /// The search stopped after `Settings::max_changes` active-set changes. The solution satisfies the levels
    /// that were finished, but not necessarily the rest.
    MaxChanges,
    /// The problem was accepted and has not been solved since.
    Ready,
    /// No problem has been set.
    NoProblem,
    /// The sizes of the matrix, bounds or reference do not match each other or the solver's capacity.
    InvalidDimensions,
    /// The level ends are empty, decreasing, too many, or do not end at the number of rows.
    InvalidLevels,
    /// A lower bound is above its upper bound, is NaN, or the bounds leave no value (both infinite on one
    /// side).
    InvalidBounds,
    /// The matrix or the reference has a non-finite entry.
    InvalidData,
    /// The metric is not symmetric positive definite.
    InvalidMetric,
};

/// Whether a status means the solver holds a solution (possibly incomplete, for `Status::MaxChanges`).
constexpr bool has_solution(Status status) noexcept {
    return status == Status::Solved || status == Status::MaxChanges;
}

/// Name of a status, for messages.
constexpr const char* to_string(Status status) noexcept {
    switch (status) {
        case Status::Solved:
            return "solved";
        case Status::MaxChanges:
            return "maximum number of active-set changes reached";
        case Status::Ready:
            return "ready";
        case Status::NoProblem:
            return "no problem set";
        case Status::InvalidDimensions:
            return "invalid dimensions";
        case Status::InvalidLevels:
            return "invalid level ends";
        case Status::InvalidBounds:
            return "invalid bounds";
        case Status::InvalidData:
            return "non-finite data";
        case Status::InvalidMetric:
            return "metric not symmetric positive definite";
    }
    return "unknown";
}

}  // namespace hlq
