#pragma once

/// @file
/// The hierarchical least-squares solver.

#include <algorithm>
#include <cmath>
#include <limits>
#include <type_traits>

#include <Eigen/Dense>

#include <hlq/detail/priority_lq.hpp>
#include <hlq/settings.hpp>
#include <hlq/status.hpp>

namespace hlq {

/// How a constraint stands in the active set at the solution.
enum class Activity {
    /// Not in the active set.
    Inactive,
    /// Held at its lower bound.
    Lower,
    /// Held at its upper bound.
    Upper,
    /// An equality (lower == upper), always active.
    Equality,
};

/**
 * Solver of hierarchical (lexicographic) least-squares problems with inequality constraints.
 *
 * A problem is a stack of levels; level k has rows A_k and bounds l_k <= A_k x <= u_k (an equality where l ==
 * u). Level 1 minimizes the squared violation of its bounds; each following level minimizes its own among the
 * minimizers of the levels above. Among the points that solve every level, the solver returns the one closest
 * to a reference (zero by default) in the metric (the identity by default).
 *
 * All active rows are kept in one factorization, Gram-Schmidt in priority order (see `detail::PriorityLQ`),
 * and the active set is searched with a primal method: a ratio test keeps the iterate feasible for the levels
 * already solved, and multipliers of each level's subproblem decide which constraints to release.
 *
 * Memory: the constructor allocates all workspace; `solve` never allocates. With fixed capacities
 * (`MaxRows`, `MaxCols`, `MaxLevels`) nothing is allocated on the heap at all.
 *
 * @tparam Scalar    float or double
 * @tparam MaxRows   compile-time capacity of rows, or Eigen::Dynamic
 * @tparam MaxCols   compile-time capacity of variables, or Eigen::Dynamic
 * @tparam MaxLevels compile-time capacity of levels, or Eigen::Dynamic
 */
template <typename Scalar = double, int MaxRows = Eigen::Dynamic, int MaxCols = Eigen::Dynamic,
          int MaxLevels = Eigen::Dynamic>
class Solver {
    static_assert(std::is_floating_point_v<Scalar>, "Scalar must be a floating-point type");

public:
    using Index = Eigen::Index;
    /// Vector over variables.
    using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1, 0, MaxCols, 1>;
    /// Vector over constraints.
    using ConstraintVector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1, 0, MaxRows, 1>;
    /// Constraint matrix, row major.
    using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, MaxRows, MaxCols>;
    /// Square matrix over variables.
    using MetricMatrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, 0, MaxCols, MaxCols>;
    /// End of each level.
    using LevelEnds = Eigen::Matrix<Index, Eigen::Dynamic, 1, 0, MaxLevels, 1>;

    /**
     * Allocates a solver for problems with up to `max_rows` constraints and exactly `cols` variables.
     *
     * With compile-time capacities, `max_rows` and `cols` must not exceed them.
     */
    Solver(Index max_rows, Index cols)
        : max_rows_(max_rows),
          cols_(cols),
          matrix_(max_rows, cols),
          scaled_(max_rows, cols),
          lower_(max_rows),
          upper_(max_rows),
          metric_inv_(cols, cols),
          reference_(cols),
          active_(max_rows),
          lower_side_(max_rows),
          locked_(max_rows),
          equality_(max_rows),
          lq_(max_rows, cols),
          goal_(max_rows),
          residual_(max_rows),
          dual_(max_rows),
          values_(max_rows),
          at_reference_(max_rows),
          violation_(max_rows),
          multipliers_(max_rows),
          step_(cols),
          primal_(cols),
          point_(cols),
          solution_(cols) {
        eigen_assert(max_rows >= 0 && cols >= 0);
        eigen_assert(MaxRows == Eigen::Dynamic || max_rows <= MaxRows);
        eigen_assert(MaxCols == Eigen::Dynamic || cols <= MaxCols);
        metric_inv_.setIdentity();
        reference_.setZero();
        solution_.setZero();
        active_.setZero();
        lower_side_.setZero();
        locked_.setZero();
        equality_.setZero();
    }

    /// A solver of the compile-time capacities.
    Solver()
        requires(MaxRows != Eigen::Dynamic && MaxCols != Eigen::Dynamic)
        : Solver(MaxRows, MaxCols) {}

    /**
     * Sets the problem: rows `matrix`, bounds `lower <= matrix x <= upper`, and the end of each level.
     *
     * Level k is rows `[level_ends(k-1), level_ends(k))`, with `level_ends(-1) = 0`; the last end is the
     * number of rows. Bounds may be infinite. When the number of rows and the level ends match the previous
     * problem, its active set is kept as a warm start.
     *
     * @return `Status::Ready`, or the reason the problem was rejected (the previous one is then lost)
     */
    template <typename MatrixType, typename LowerType, typename UpperType, typename EndsType>
    Status set_problem(const Eigen::MatrixBase<MatrixType>& matrix, const Eigen::MatrixBase<LowerType>& lower,
                       const Eigen::MatrixBase<UpperType>& upper,
                       const Eigen::MatrixBase<EndsType>& level_ends) {
        has_problem_ = false;
        status_ = validate(matrix, lower, upper, level_ends);
        if (status_ != Status::Ready) {
            return status_;
        }

        const Index rows = matrix.rows();
        const Index levels = level_ends.size();
        bool same = rows == rows_ && levels == levels_;
        for (Index k = 0; same && k < levels; ++k) {
            same = static_cast<Index>(level_ends(k)) == level_ends_(k);
        }
        if (!same) {
            level_ends_.resize(levels);
            for (Index k = 0; k < levels; ++k) {
                level_ends_(k) = static_cast<Index>(level_ends(k));
            }
            lq_.prepare(levels);
            noise_.resize(levels);
            rows_ = rows;
            levels_ = levels;
            reset();
        }
        matrix_.topRows(rows) = matrix.template cast<Scalar>();
        lower_.head(rows) = lower.template cast<Scalar>();
        upper_.head(rows) = upper.template cast<Scalar>();

        // Equalities are always active. A warm-start constraint held at a bound that is now infinite is
        // released.
        equality_.head(rows) = lower_.head(rows).array() == upper_.head(rows).array();
        for (Index i = 0; i < rows; ++i) {
            if (equality_(i)) {
                active_(i) = true;
                lower_side_(i) = true;
            } else if (active_(i) && !std::isfinite(lower_side_(i) ? lower_(i) : upper_(i))) {
                active_(i) = false;
            }
        }
        has_problem_ = true;
        scaled_valid_ = false;
        return status_;
    }

    /**
     * Sets the metric W: among the points that solve every level, the solution minimizes (x - r)^T W (x - r),
     * with r the reference. W must be symmetric positive definite.
     */
    template <typename MetricType>
    Status set_metric(const Eigen::MatrixBase<MetricType>& metric) {
        if (metric.rows() != cols_ || metric.cols() != cols_) {
            return Status::InvalidDimensions;
        }
        const MetricMatrix w = metric.template cast<Scalar>();
        if (!w.allFinite() || !w.isApprox(w.transpose(), settings_.rank_tolerance)) {
            return Status::InvalidMetric;
        }
        const Eigen::LLT<MetricMatrix> llt(w);
        if (llt.info() != Eigen::Success) {
            return Status::InvalidMetric;
        }
        // x = C z with C = U^-1 and W = U^T U, so that the metric becomes the identity in z.
        metric_inv_.setIdentity();
        llt.matrixU().solveInPlace(metric_inv_);
        identity_metric_ = metric_inv_.isIdentity(Scalar(0));
        scaled_valid_ = false;
        return has_problem_ ? (status_ = Status::Ready) : Status::NoProblem;
    }

    /// Sets the reference: among the points that solve every level, the solution is the closest to it.
    template <typename ReferenceType>
    Status set_reference(const Eigen::MatrixBase<ReferenceType>& reference) {
        if (reference.size() != cols_) {
            return Status::InvalidDimensions;
        }
        if (!reference.allFinite()) {
            return Status::InvalidData;
        }
        reference_ = reference.template cast<Scalar>();
        return has_problem_ ? (status_ = Status::Ready) : Status::NoProblem;
    }

    /// Forgets the active set, so that the next solve starts from the equalities only.
    void reset() noexcept {
        active_ = equality_;
        lower_side_ = equality_;
        locked_.setZero();
    }

    /// Solves the problem; never allocates.
    Status solve() noexcept {
        if (!has_problem_) {
            return status_;
        }
        prepare_rows();
        changes_ = 0;
        locked_.head(rows_) = equality_.head(rows_);
        lq_.factor(factored_rows(), active_, level_ends_, noise_);
        changed_ = 0;
        forward();
        point_ = primal_;

        const bool equalities_only = equality_.head(rows_).all();
        // Phases 0 to levels - 1 solve the levels; phase `levels` chooses among their optima the point
        // closest to the reference, releasing the constraints that hold the step back and that no level
        // needs.
        for (Index h = 0; !equalities_only && changes_ < settings_.max_changes && h <= levels_; ++h) {
            phase_ = h;
            if (h < levels_) {
                // A level left out of the factorization during an earlier phase comes back now.
                const bool appended = lq_.factored() == h;
                if (appended) {
                    lq_.append(factored_rows(), active_, h == 0 ? 0 : level_ends_(h - 1), level_ends_(h),
                               noise_);
                    changed_ = std::min(changed_, h);
                }
                if (activate_violated(h) > 0 || appended) {
                    forward();
                }
            }
            while (changes_ < settings_.max_changes) {
                // Add the first constraint that blocks the step from the iterate to the active set's
                // solution.
                Index row = -1;
                bool at_lower = false;
                const Scalar step = ratio_test(h, row, at_lower);
                if (row >= 0) {
                    point_ += step * (primal_ - point_);
                    activate(row, at_lower);
                    forward();
                    ++changes_;
                    continue;
                }
                point_ = primal_;

                // Release the constraint whose multiplier has the most wrong sign.
                lq_.multipliers(h, residual_, dual_);
                const Index active_above = h < levels_ ? lq_.begin(h + 1) : lq_.size();
                // Relative: rounding noise in a multiplier scales with the largest of the subproblem.
                const Scalar zero =
                    settings_.dual_tolerance
                    * (Scalar(1)
                       + (active_above > 0 ? dual_.head(active_above).cwiseAbs().maxCoeff() : Scalar(0)));
                Index drop = -1;
                Scalar worst = zero;
                for (Index p = 0; p < active_above; ++p) {
                    const Index i = lq_.row(p);
                    const Scalar wrong = lower_side_(i) ? dual_(p) : -dual_(p);
                    if (!locked_(i) && wrong > worst) {
                        worst = wrong;
                        drop = i;
                    }
                }
                if (drop >= 0) {
                    deactivate(drop);
                    forward();
                    ++changes_;
                    continue;
                }

                // A strictly right multiplier keeps its constraint at its bound for every lower level.
                for (Index p = 0; p < active_above; ++p) {
                    const Index i = lq_.row(p);
                    if ((lower_side_(i) ? dual_(p) : -dual_(p)) < -zero) {
                        locked_(i) = true;
                    }
                }
                break;
            }
        }

        if (lq_.factored() < levels_) {
            // Stopped early: the accessors need every level.
            while (lq_.factored() < levels_) {
                const Index k = lq_.factored();
                lq_.append(factored_rows(), active_, k == 0 ? 0 : level_ends_(k - 1), level_ends_(k), noise_);
                changed_ = std::min(changed_, k);
            }
            forward();
        }
        solution_ = point_;
        auto values = values_.head(rows_);
        values.noalias() = matrix_.topRows(rows_) * solution_;
        violation_.head(rows_) = values - values.cwiseMax(lower_.head(rows_)).cwiseMin(upper_.head(rows_));
        status_ = changes_ < settings_.max_changes ? Status::Solved : Status::MaxChanges;
        return status_;
    }

    /// Tolerances and limits; changes apply from the next solve.
    [[nodiscard]] Settings<Scalar>& settings() noexcept { return settings_; }
    [[nodiscard]] const Settings<Scalar>& settings() const noexcept { return settings_; }

    /// Outcome of the last call that set up or solved the problem.
    [[nodiscard]] Status status() const noexcept { return status_; }
    /// The solution of the last solve.
    [[nodiscard]] const Vector& solution() const noexcept { return solution_; }
    /// Violation of each bound at the solution: A x minus A x clamped to [lower, upper].
    [[nodiscard]] auto violation() const noexcept { return violation_.head(rows_); }
    /// Squared violation of a level at the solution: what that level minimizes.
    [[nodiscard]] Scalar level_cost(Index level) const noexcept {
        const Index begin = level == 0 ? 0 : level_ends_(level - 1);
        return violation_.segment(begin, level_ends_(level) - begin).squaredNorm();
    }
    /// Number of active-set changes in the last solve.
    [[nodiscard]] Index changes() const noexcept { return changes_; }
    /// Number of independent active rows of a level at the solution.
    [[nodiscard]] Index rank(Index level) const noexcept { return lq_.rank(level); }
    /// How a constraint stands in the active set at the solution.
    [[nodiscard]] Activity activity(Index row) const noexcept {
        if (equality_(row)) {
            return Activity::Equality;
        }
        if (!active_(row)) {
            return Activity::Inactive;
        }
        return lower_side_(row) ? Activity::Lower : Activity::Upper;
    }

    /**
     * Multipliers of the subproblem of a level at the solution: for its own rows, their residuals A x - b;
     * for the active rows of the levels above, the multipliers that balance them (the minimum-norm ones
     * within each level); zero for the other rows. A constraint at its upper bound has a nonnegative
     * multiplier and one at its lower bound a nonpositive one. `level == levels()` gives those of the final
     * choice among the optima (the distance to the reference). Valid after `Status::Solved`.
     */
    [[nodiscard]] auto multipliers(Index level) noexcept {
        eigen_assert(level >= 0 && level <= levels_);
        lq_.multipliers(level, residual_, dual_);
        multipliers_.head(rows_).setZero();
        const Index active_above = level < levels_ ? lq_.begin(level + 1) : lq_.size();
        for (Index p = 0; p < active_above; ++p) {
            multipliers_(lq_.row(p)) = dual_(p);
        }
        return multipliers_.head(rows_);
    }

    /// Number of constraints of the current problem.
    [[nodiscard]] Index rows() const noexcept { return rows_; }
    /// Number of variables.
    [[nodiscard]] Index cols() const noexcept { return cols_; }
    /// Number of levels of the current problem.
    [[nodiscard]] Index levels() const noexcept { return levels_; }
    /// Capacity of constraints.
    [[nodiscard]] Index max_rows() const noexcept { return max_rows_; }

private:
    using Factorization = detail::PriorityLQ<Scalar, MaxRows, MaxCols, MaxLevels>;
    using Mask = Eigen::Array<bool, Eigen::Dynamic, 1, 0, MaxRows, 1>;
    using LevelsVector = typename Factorization::LevelsVector;

    template <typename MatrixType, typename LowerType, typename UpperType, typename EndsType>
    Status validate(const Eigen::MatrixBase<MatrixType>& matrix, const Eigen::MatrixBase<LowerType>& lower,
                    const Eigen::MatrixBase<UpperType>& upper,
                    const Eigen::MatrixBase<EndsType>& level_ends) const {
        const Index rows = matrix.rows();
        if (matrix.cols() != cols_ || rows > max_rows_ || lower.size() != rows || upper.size() != rows) {
            return Status::InvalidDimensions;
        }
        const Index levels = level_ends.size();
        if (levels == 0 || (MaxLevels != Eigen::Dynamic && levels > MaxLevels)) {
            return Status::InvalidLevels;
        }
        for (Index k = 0, previous = 0; k < levels; ++k) {
            const auto end = static_cast<Index>(level_ends(k));
            if (end < previous) {
                return Status::InvalidLevels;
            }
            previous = end;
        }
        if (static_cast<Index>(level_ends(levels - 1)) != rows) {
            return Status::InvalidLevels;
        }
        constexpr Scalar infinity = std::numeric_limits<Scalar>::infinity();
        for (Index i = 0; i < rows; ++i) {
            const auto l = static_cast<Scalar>(lower(i));
            const auto u = static_cast<Scalar>(upper(i));
            if (!(l <= u) || l == infinity || u == -infinity) {
                return Status::InvalidBounds;
            }
        }
        if (!matrix.allFinite()) {
            return Status::InvalidData;
        }
        return Status::Ready;
    }

    /// The rows that are factored: the rows of the problem in the coordinates of the metric.
    [[nodiscard]] const Matrix& factored_rows() const noexcept {
        return identity_metric_ ? matrix_ : scaled_;
    }

    /// Rows in the coordinates of the metric, and the rank threshold of each level.
    void prepare_rows() noexcept {
        if (!scaled_valid_ && !identity_metric_) {
            scaled_.topRows(rows_).noalias() = matrix_.topRows(rows_) * metric_inv_;
        }
        scaled_valid_ = true;
        at_reference_.head(rows_).noalias() = matrix_.topRows(rows_) * reference_;
        // Absolute: rounding noise scales with the longest row of a level, not with what is left of a row.
        for (Index k = 0, begin = 0; k < levels_; ++k) {
            const Index end = level_ends_(k);
            const auto level = factored_rows().middleRows(begin, end - begin);
            noise_(k) =
                end > begin ? settings_.rank_tolerance * level.rowwise().norm().maxCoeff() : Scalar(0);
            begin = end;
        }
    }

    /// Solution of the active set, and the residual of each active row.
    void forward() noexcept {
        const Index size = lq_.size();
        for (Index p = lq_.begin(changed_); p < size; ++p) {
            const Index i = lq_.row(p);
            goal_(p) = (lower_side_(i) ? lower_(i) : upper_(i)) - at_reference_(i);
        }
        lq_.solve(goal_, residual_, step_, changed_);
        changed_ = levels_;
        primal_ = reference_;
        if (identity_metric_) {
            primal_ += step_;
        } else {
            primal_.noalias() += metric_inv_ * step_;
        }
    }

    /// Phase h needs only levels <= h: the levels below leave the factorization before the first change,
    /// which then no longer updates their rows, and come back in their own phase.
    void trim() noexcept {
        if (phase_ + 1 < lq_.factored()) {
            lq_.truncate(phase_);
        }
    }

    void activate(Index row, bool at_lower) noexcept {
        trim();
        active_(row) = true;
        lower_side_(row) = at_lower;
        const Index level = level_of(row);
        lq_.insert(factored_rows(), row, level, noise_);
        changed_ = std::min(changed_, level);
    }

    void deactivate(Index row) noexcept {
        trim();
        active_(row) = false;
        locked_(row) = false;
        changed_ = std::min(changed_, level_of(row));
        lq_.remove(row, noise_);
    }

    /// Activates the constraints of level h that the iterate violates; returns how many.
    Index activate_violated(Index h) noexcept {
        const Index begin = h == 0 ? 0 : level_ends_(h - 1);
        const Index end = level_ends_(h);
        values_.segment(begin, end - begin).noalias() = matrix_.middleRows(begin, end - begin) * point_;
        Index count = 0;
        for (Index i = begin; i < end && changes_ < settings_.max_changes; ++i) {
            if (active_(i)) {
                continue;
            }
            const bool below = lower_(i) - values_(i) > settings_.feasibility_tolerance;
            if (below || values_(i) - upper_(i) > settings_.feasibility_tolerance) {
                activate(i, below);
                ++changes_;
                ++count;
            }
        }
        return count;
    }

    /// Step from the iterate towards the active set's solution, up to the first inactive constraint of a
    /// level
    /// <= h (every level, in the final phase) that it would violate; that constraint and its bound are
    /// returned in `row` and `at_lower`.
    Scalar ratio_test(Index h, Index& row, bool& at_lower) noexcept {
        Scalar step = 1;
        row = -1;
        const Index end = level_ends_(std::min(h, levels_ - 1));
        values_.head(end).noalias() = matrix_.topRows(end) * primal_;
        for (Index i = 0; i < end; ++i) {
            if (active_(i)) {
                continue;
            }
            const bool above = values_(i) - upper_(i) > settings_.feasibility_tolerance;
            if (above || lower_(i) - values_(i) > settings_.feasibility_tolerance) {
                // The iterate satisfies the bound, so the denominator is nonzero.
                const Scalar from = matrix_.row(i).dot(point_);
                const Scalar bound = above ? upper_(i) : lower_(i);
                const Scalar t = std::clamp((bound - from) / (values_(i) - from), Scalar(0), Scalar(1));
                if (row < 0 || t < step) {
                    step = t;
                    row = i;
                    at_lower = !above;
                }
            }
        }
        return step;
    }

    [[nodiscard]] Index level_of(Index row) const noexcept {
        Index k = 0;
        while (level_ends_(k) <= row) {
            ++k;
        }
        return k;
    }

    Index max_rows_;
    Index cols_;
    Index rows_ = 0;
    Index levels_ = 0;
    bool has_problem_ = false;
    bool scaled_valid_ = false;
    bool identity_metric_ = true;
    Index phase_ = 0;    // level being solved, or levels_ for the choice among the optima
    Index changed_ = 0;  // first level whose active rows changed since the last forward solve
    Status status_ = Status::NoProblem;
    Settings<Scalar> settings_;
    Index changes_ = 0;

    Matrix matrix_;  // rows of the problem
    Matrix scaled_;  // rows times metric_inv_: the rows that are factored, unless the metric is the identity
    ConstraintVector lower_;
    ConstraintVector upper_;
    LevelEnds level_ends_;
    LevelsVector noise_;       // absolute rank threshold of each level
    MetricMatrix metric_inv_;  // x = metric_inv_ z
    Vector reference_;

    Mask active_;
    Mask lower_side_;  // active constraints held at their lower bound
    Mask locked_;      // active constraints that may no longer leave
    Mask equality_;

    Factorization lq_;
    ConstraintVector goal_;
    ConstraintVector residual_;
    ConstraintVector dual_;
    ConstraintVector values_;
    ConstraintVector at_reference_;  // A times the reference
    ConstraintVector violation_;
    ConstraintVector multipliers_;
    Vector step_;
    Vector primal_;  // solution of the active set
    Vector point_;   // iterate of the search
    Vector solution_;
};

}  // namespace hlq
