#pragma once

/// @file
/// The hierarchical least-squares solver.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <limits>
#include <span>
#include <type_traits>

#include <hlq/detail/dense.hpp>
#include <hlq/detail/priority_lq.hpp>
#include <hlq/settings.hpp>
#include <hlq/status.hpp>
#include <hlq/types.hpp>
#include <hlq/view.hpp>

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
 * Data is read from any `MatrixInput` and `VectorInput` (Eigen types, standard containers, `MatrixView`) and
 * copied; results are `std::span`s into the solver, valid until its next call that sets up or solves.
 *
 * Memory: the constructor allocates all workspace; `solve` never allocates. With fixed capacities
 * (`MaxRows`, `MaxCols`, `MaxLevels`) nothing is allocated on the heap at all.
 *
 * @tparam Scalar    float or double
 * @tparam MaxRows   compile-time capacity of rows, or `hlq::dynamic`
 * @tparam MaxCols   compile-time capacity of variables, or `hlq::dynamic`
 * @tparam MaxLevels compile-time capacity of levels, or `hlq::dynamic`
 */
template <typename Scalar = double, int MaxRows = dynamic, int MaxCols = dynamic, int MaxLevels = dynamic>
class Solver {
    static_assert(std::is_floating_point_v<Scalar>, "Scalar must be a floating-point type");

public:
    using Index = hlq::Index;

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
        HLQ_ASSERT(max_rows >= 0 && cols >= 0);
        HLQ_ASSERT(MaxRows == dynamic || max_rows <= MaxRows);
        HLQ_ASSERT(MaxCols == dynamic || cols <= MaxCols);
        metric_inv_.fill(Scalar(0));
        for (Index j = 0; j < cols; ++j) {
            metric_inv_(j, j) = Scalar(1);
        }
        reference_.fill(Scalar(0));
        solution_.fill(Scalar(0));
        active_.fill(false);
        lower_side_.fill(false);
        locked_.fill(false);
        equality_.fill(false);
    }

    /// A solver of the compile-time capacities.
    Solver()
        requires(MaxRows != dynamic && MaxCols != dynamic)
        : Solver(MaxRows, MaxCols) {}

    /**
     * Sets the problem: rows `matrix`, bounds `lower <= matrix x <= upper`, and the end of each level.
     *
     * Level k is rows `[level_ends[k-1], level_ends[k])`, with `level_ends[-1] = 0`; the last end is the
     * number of rows. Bounds may be infinite. When the number of rows and the level ends match the previous
     * problem, its active set is kept as a warm start.
     *
     * @return `Status::Ready`, or the reason the problem was rejected (the previous one is then lost)
     */
    template <MatrixInput MatrixType, VectorInput LowerType, VectorInput UpperType, VectorInput EndsType>
    Status set_problem(const MatrixType& matrix, const LowerType& lower, const UpperType& upper,
                       const EndsType& level_ends) {
        has_problem_ = false;
        status_ = validate(matrix, lower, upper, level_ends);
        if (status_ != Status::Ready) {
            return status_;
        }

        const auto rows = static_cast<Index>(matrix.rows());
        const Index levels = detail::length(level_ends);
        bool same = rows == rows_ && levels == levels_;
        for (Index k = 0; same && k < levels; ++k) {
            same = static_cast<Index>(detail::element(level_ends, k)) == level_ends_(k);
        }
        if (!same) {
            level_ends_.resize(levels);
            for (Index k = 0; k < levels; ++k) {
                level_ends_(k) = static_cast<Index>(detail::element(level_ends, k));
            }
            lq_.prepare(levels);
            noise_.resize(levels);
            rows_ = rows;
            levels_ = levels;
            reset();
        }
        for (Index i = 0; i < rows; ++i) {
            for (Index j = 0; j < cols_; ++j) {
                matrix_(i, j) = static_cast<Scalar>(matrix(i, j));
            }
            lower_(i) = static_cast<Scalar>(detail::element(lower, i));
            upper_(i) = static_cast<Scalar>(detail::element(upper, i));
        }

        // Equalities are always active. A warm-start constraint held at a bound that is now infinite is
        // released.
        for (Index i = 0; i < rows; ++i) {
            equality_(i) = lower_(i) == upper_(i);
            if (equality_(i)) {
                active_(i) = true;
                lower_side_(i) = true;
            } else if (active_(i) && !std::isfinite(lower_side_(i) ? lower_(i) : upper_(i))) {
                active_(i) = false;
            }
        }
        // A last level that may be folded gets the fold's storage now, as solve() does not allocate.
        const Index last = levels_ < 2 ? 0 : level_ends_(levels_ - 2);
        bool foldable = rows_ - last >= cols_;
        for (Index i = last; foldable && i < rows_; ++i) {
            foldable = equality_(i);
        }
        if (foldable) {
            fold_inv_.resize(cols_, cols_);
            fold_reference_.resize(cols_);
        }
        has_problem_ = true;
        scaled_valid_ = false;
        return status_;
    }

    /// `set_problem` with the level ends given in braces: `solver.set_problem(a, lower, upper, {2, 5})`.
    template <MatrixInput MatrixType, VectorInput LowerType, VectorInput UpperType>
    Status set_problem(const MatrixType& matrix, const LowerType& lower, const UpperType& upper,
                       std::initializer_list<Index> level_ends) {
        return set_problem(matrix, lower, upper,
                           std::span<const Index>(level_ends.begin(), level_ends.size()));
    }

    /**
     * Sets the metric W: among the points that solve every level, the solution minimizes (x - r)^T W (x - r),
     * with r the reference. W must be symmetric positive definite.
     */
    template <MatrixInput MetricType>
    Status set_metric(const MetricType& metric) {
        if (static_cast<Index>(metric.rows()) != cols_ || static_cast<Index>(metric.cols()) != cols_) {
            return Status::InvalidDimensions;
        }
        Metric w(cols_, cols_);
        Scalar asymmetry = 0;
        Scalar size = 0;
        for (Index i = 0; i < cols_; ++i) {
            for (Index j = 0; j < cols_; ++j) {
                w(i, j) = static_cast<Scalar>(metric(i, j));
                if (!std::isfinite(w(i, j))) {
                    return Status::InvalidMetric;
                }
                size += w(i, j) * w(i, j);
            }
        }
        for (Index i = 0; i < cols_; ++i) {
            for (Index j = 0; j < cols_; ++j) {
                asymmetry += (w(i, j) - w(j, i)) * (w(i, j) - w(j, i));
            }
        }
        if (asymmetry > settings_.rank_tolerance * settings_.rank_tolerance * size
            || !detail::cholesky(w.view())) {
            return Status::InvalidMetric;
        }
        // x = C z with C = U^-1 and W = U^T U (U = L^T), so that the metric becomes the identity in z. Row i
        // of C is column i of L^-1, the solution of L c = e_i.
        ColsVector column(cols_);
        for (Index i = 0; i < cols_; ++i) {
            column.fill(Scalar(0));
            column(i) = Scalar(1);
            detail::solve_lower(w.view(), column.data());
            std::copy_n(column.data(), cols_, metric_inv_.row(i));
        }
        identity_metric_ = true;
        for (Index i = 0; i < cols_ && identity_metric_; ++i) {
            for (Index j = 0; j < cols_; ++j) {
                if (metric_inv_(i, j) != (i == j ? Scalar(1) : Scalar(0))) {
                    identity_metric_ = false;
                    break;
                }
            }
        }
        scaled_valid_ = false;
        return has_problem_ ? (status_ = Status::Ready) : Status::NoProblem;
    }

    /// Sets the reference: among the points that solve every level, the solution is the closest to it.
    template <VectorInput ReferenceType>
    Status set_reference(const ReferenceType& reference) {
        if (detail::length(reference) != cols_) {
            return Status::InvalidDimensions;
        }
        for (Index j = 0; j < cols_; ++j) {
            if (!std::isfinite(static_cast<Scalar>(detail::element(reference, j)))) {
                return Status::InvalidData;
            }
        }
        for (Index j = 0; j < cols_; ++j) {
            reference_(j) = static_cast<Scalar>(detail::element(reference, j));
        }
        return has_problem_ ? (status_ = Status::Ready) : Status::NoProblem;
    }

    /// Forgets the active set, so that the next solve starts from the equalities only.
    void reset() noexcept {
        for (Index i = 0; i < max_rows_; ++i) {
            active_(i) = equality_(i);
            lower_side_(i) = equality_(i);
            locked_(i) = false;
        }
    }

    /// Solves the problem; never allocates.
    Status solve() noexcept {
        if (!has_problem_) {
            return status_;
        }
        prepare_rows();
        changes_ = 0;
        bool equalities_only = true;
        for (Index i = 0; i < rows_; ++i) {
            locked_(i) = equality_(i);
            equalities_only = equalities_only && equality_(i);
        }
        lq_.factor(factored_rows(), active_, level_ends_, noise_, searched_);
        changed_ = 0;
        forward();
        copy(primal_, point_);

        // Phases 0 to searched - 1 solve the levels; phase `searched` chooses among their optima the point
        // closest to the reference, releasing the constraints that hold the step back and that no level
        // needs. With a folded last level, that choice is the last level itself (see `fold`).
        for (Index h = 0; !equalities_only && changes_ < settings_.max_changes && h <= searched_; ++h) {
            phase_ = h;
            if (h < searched_) {
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
                    for (Index j = 0; j < cols_; ++j) {
                        point_(j) += step * (primal_(j) - point_(j));
                    }
                    activate(row, at_lower);
                    forward();
                    ++changes_;
                    continue;
                }
                copy(primal_, point_);

                // Release the constraint whose multiplier has the most wrong sign.
                lq_.multipliers(h, residual_, dual_);
                const Index active_above = h < searched_ ? lq_.begin(h + 1) : lq_.size();
                // Relative: rounding noise in a multiplier scales with the largest of the subproblem.
                Scalar largest = 0;
                for (Index p = 0; p < active_above; ++p) {
                    largest = std::max(largest, std::abs(dual_(p)));
                }
                const Scalar zero = settings_.dual_tolerance * (Scalar(1) + largest);
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

        if (lq_.factored() < searched_) {
            // Stopped early: the accessors need every level.
            while (lq_.factored() < searched_) {
                const Index k = lq_.factored();
                lq_.append(factored_rows(), active_, k == 0 ? 0 : level_ends_(k - 1), level_ends_(k), noise_);
                changed_ = std::min(changed_, k);
            }
            forward();
        }
        copy(point_, solution_);
        detail::multiply(matrix_.block(0, 0, rows_, cols_), solution_.data(), values_.data());
        for (Index i = 0; i < rows_; ++i) {
            violation_(i) = values_(i) - std::min(std::max(values_(i), lower_(i)), upper_(i));
        }
        status_ = changes_ < settings_.max_changes ? Status::Solved : Status::MaxChanges;
        return status_;
    }

    /// Tolerances and limits; changes apply from the next solve.
    [[nodiscard]] Settings<Scalar>& settings() noexcept { return settings_; }
    [[nodiscard]] const Settings<Scalar>& settings() const noexcept { return settings_; }

    /// Outcome of the last call that set up or solved the problem.
    [[nodiscard]] Status status() const noexcept { return status_; }
    /// The solution of the last solve.
    [[nodiscard]] std::span<const Scalar> solution() const noexcept { return span(solution_, cols_); }
    /// Violation of each bound at the solution: A x minus A x clamped to [lower, upper].
    [[nodiscard]] std::span<const Scalar> violation() const noexcept { return span(violation_, rows_); }
    /// Squared violation of a level at the solution: what that level minimizes.
    [[nodiscard]] Scalar level_cost(Index level) const noexcept {
        const Index begin = level == 0 ? 0 : level_ends_(level - 1);
        return detail::squared_norm(violation_.data() + begin, level_ends_(level) - begin);
    }
    /// Number of active-set changes in the last solve.
    [[nodiscard]] Index changes() const noexcept { return changes_; }
    /// Number of independent active rows of a level at the solution.
    [[nodiscard]] Index rank(Index level) const noexcept {
        if (fold_ && level == searched_) {
            // A folded last level takes every direction that the levels above leave.
            Index above = 0;
            for (Index k = 0; k < searched_; ++k) {
                above += lq_.rank(k);
            }
            return cols_ - above;
        }
        return lq_.rank(level);
    }
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
    [[nodiscard]] std::span<const Scalar> multipliers(Index level) noexcept {
        HLQ_ASSERT(level >= 0 && level <= levels_);
        std::fill_n(multipliers_.data(), rows_, Scalar(0));
        if (fold_ && level == levels_) {
            multipliers_after_fold();
            return span(multipliers_, rows_);
        }
        if (lq_.factored() > searched_) {
            lq_.truncate(searched_ - 1);  // left there by multipliers(levels())
        }
        // With a folded last level, its subproblem is the choice among the optima of the levels above.
        const Index h = std::min(level, searched_);
        lq_.multipliers(h, residual_, dual_);
        const Index active_above = h < searched_ ? lq_.begin(h + 1) : lq_.size();
        for (Index p = 0; p < active_above; ++p) {
            multipliers_(lq_.row(p)) = dual_(p);
        }
        if (fold_ && level == searched_) {
            // Its own rows: their residuals, A x - b.
            for (Index i = level == 0 ? 0 : level_ends_(level - 1); i < rows_; ++i) {
                multipliers_(i) = violation_(i);
            }
        }
        return span(multipliers_, rows_);
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
    using Rows = detail::Matrix<Scalar, MaxRows, MaxCols>;
    using Metric = detail::Matrix<Scalar, MaxCols, MaxCols>;
    using ColsVector = detail::Vector<Scalar, MaxCols>;
    using RowsVector = detail::Vector<Scalar, MaxRows>;
    using Mask = detail::Vector<bool, MaxRows>;
    using LevelEnds = detail::Vector<Index, MaxLevels>;
    using LevelsVector = typename Factorization::LevelsVector;

    template <typename V>
    [[nodiscard]] static std::span<const Scalar> span(const V& v, Index size) noexcept {
        return {v.data(), static_cast<std::size_t>(size)};
    }

    void copy(const ColsVector& from, ColsVector& to) noexcept { std::copy_n(from.data(), cols_, to.data()); }

    template <typename MatrixType, typename LowerType, typename UpperType, typename EndsType>
    Status validate(const MatrixType& matrix, const LowerType& lower, const UpperType& upper,
                    const EndsType& level_ends) const {
        const auto rows = static_cast<Index>(matrix.rows());
        if (static_cast<Index>(matrix.cols()) != cols_ || rows > max_rows_ || detail::length(lower) != rows
            || detail::length(upper) != rows) {
            return Status::InvalidDimensions;
        }
        const Index levels = detail::length(level_ends);
        if (levels == 0 || (MaxLevels != dynamic && levels > MaxLevels)) {
            return Status::InvalidLevels;
        }
        for (Index k = 0, previous = 0; k < levels; ++k) {
            const auto end = static_cast<Index>(detail::element(level_ends, k));
            if (end < previous) {
                return Status::InvalidLevels;
            }
            previous = end;
        }
        if (static_cast<Index>(detail::element(level_ends, levels - 1)) != rows) {
            return Status::InvalidLevels;
        }
        constexpr Scalar infinity = std::numeric_limits<Scalar>::infinity();
        for (Index i = 0; i < rows; ++i) {
            const auto l = static_cast<Scalar>(detail::element(lower, i));
            const auto u = static_cast<Scalar>(detail::element(upper, i));
            if (!(l <= u) || l == infinity || u == -infinity) {
                return Status::InvalidBounds;
            }
        }
        for (Index i = 0; i < rows; ++i) {
            for (Index j = 0; j < cols_; ++j) {
                if (!std::isfinite(static_cast<Scalar>(matrix(i, j)))) {
                    return Status::InvalidData;
                }
            }
        }
        return Status::Ready;
    }

    /// The metric (as C, x = r + C z), whether it is the identity, and the reference of the search: the
    /// user's, or those of a folded last level.
    [[nodiscard]] const Metric& metric() const noexcept { return fold_ ? fold_inv_ : metric_inv_; }
    [[nodiscard]] bool identity() const noexcept { return fold_ ? fold_identity_ : identity_metric_; }
    [[nodiscard]] const ColsVector& reference() const noexcept {
        return fold_ ? fold_reference_ : reference_;
    }

    /// The rows that are factored: the rows of the problem in the coordinates of the metric.
    [[nodiscard]] detail::MatrixRef<const Scalar> factored_rows() const noexcept {
        return (identity() ? matrix_ : scaled_).block(0, 0, rows_, cols_);
    }

    /**
     * Whether the last level is the choice among the optima. A last level of equalities F x = b whose rows
     * have full column rank leaves a single point, where ||F x - b||^2 = (x - r)^T F^T F (x - r) + constant,
     * with r = F^+ b, is least among the optima of the levels above: the solver's own choice among them, with
     * metric F^T F and reference r. That level is then not searched. With F = Q R, the metric is C = R^-1 and
     * r = R^-1 Q^T b. Conditioning worse than 1 / sqrt(epsilon) is left to the search.
     */
    void fold() noexcept {
        fold_ = false;
        searched_ = levels_;
        const Index begin = levels_ < 2 ? 0 : level_ends_(levels_ - 2);
        const Index count = rows_ - begin;
        if (!settings_.fold_final_level || levels_ == 0 || cols_ == 0 || count < cols_
            || fold_inv_.rows() != cols_) {
            return;
        }
        for (Index i = begin; i < rows_; ++i) {
            if (!equality_(i)) {
                return;
            }
        }
        // F^T in scaled_'s storage, which is recomputed after the fold anyway, and the reflectors' tau in
        // step_.
        const detail::MatrixRef<Scalar> ft{scaled_.row(0), cols_, count, count};
        for (Index j = 0; j < cols_; ++j) {
            for (Index i = 0; i < count; ++i) {
                ft(j, i) = matrix_(begin + i, j);
            }
        }
        Scalar* tau = step_.data();
        detail::householder_qr(ft, tau);
        Scalar largest = 0;
        Scalar smallest = std::numeric_limits<Scalar>::infinity();
        bool diagonal = true;  // R, whose transpose is the lower triangle of ft
        for (Index j = 0; j < cols_; ++j) {
            largest = std::max(largest, std::abs(ft(j, j)));
            smallest = std::min(smallest, std::abs(ft(j, j)));
            for (Index i = 0; i < j; ++i) {
                diagonal = diagonal && ft(j, i) == Scalar(0);
            }
        }
        if (!(smallest > std::sqrt(std::numeric_limits<Scalar>::epsilon()) * largest)) {
            return;
        }
        Scalar* rotated = values_.data();  // Q^T b
        for (Index i = 0; i < count; ++i) {
            rotated[i] = lower_(begin + i);
        }
        detail::apply_qt(ft, tau, rotated);
        std::copy_n(rotated, cols_, fold_reference_.data());
        detail::solve_lower_transposed(ft.block(0, 0, cols_, cols_), fold_reference_.data());
        // C = R^-1, upper triangular: column i solves the leading block of R c = e_i. R is diagonal when the
        // rows' columns are orthogonal, as for x = 0 or a weighted x = r.
        fold_inv_.fill(Scalar(0));
        fold_identity_ = diagonal;
        for (Index i = 0; i < cols_; ++i) {
            if (diagonal) {
                fold_inv_(i, i) = Scalar(1) / ft(i, i);
                fold_identity_ = fold_identity_ && ft(i, i) == Scalar(1);
                continue;
            }
            Scalar* column = primal_.data();
            std::fill_n(column, i, Scalar(0));
            column[i] = Scalar(1);
            detail::solve_lower_transposed(ft.block(0, 0, i + 1, i + 1), column);
            for (Index j = 0; j <= i; ++j) {
                fold_inv_(j, i) = column[j];
            }
        }
        fold_ = true;
        searched_ = levels_ - 1;
    }

    /// Multipliers of the user's choice among the optima, 1/2 (x - r)^T W (x - r), after a folded last level:
    /// that level joins the factorization, and the gradient is taken to its coordinates z, x = r_f + C_f z.
    void multipliers_after_fold() noexcept {
        if (lq_.factored() == searched_) {
            lq_.append(factored_rows(), active_, searched_ == 0 ? 0 : level_ends_(searched_ - 1), rows_,
                       noise_);
            changed_ = std::min(changed_, searched_);
            forward();  // for the QR of the last level; the solution stays
        }
        Scalar* g = step_.data();  // free after the solve, as is primal_
        for (Index j = 0; j < cols_; ++j) {
            g[j] = solution_(j) - reference_(j);
        }
        if (!identity_metric_) {
            // W g = U^T U g with C = U^-1 upper triangular: solve C y = g, then C^T w = y.
            for (Index i = cols_ - 1; i >= 0; --i) {
                g[i] = (g[i] - detail::dot(metric_inv_.row(i) + i + 1, g + i + 1, cols_ - i - 1))
                       / metric_inv_(i, i);
            }
            for (Index i = 0; i < cols_; ++i) {
                for (Index j = 0; j < i; ++j) {
                    g[i] -= metric_inv_(j, i) * g[j];
                }
                g[i] /= metric_inv_(i, i);
            }
        }
        Scalar* z = primal_.data();
        if (fold_identity_) {
            std::copy_n(g, cols_, z);
        } else {
            std::fill_n(z, cols_, Scalar(0));
            detail::multiply_transposed_add(Scalar(1), fold_inv_.view(), g, z);
        }
        lq_.multipliers_of(z, dual_);
        for (Index p = 0; p < lq_.size(); ++p) {
            multipliers_(lq_.row(p)) = dual_(p);
        }
    }

    /// Rows in the coordinates of the metric, and the rank threshold of each level.
    void prepare_rows() noexcept {
        if (!scaled_valid_ || fold_setting_ != settings_.fold_final_level) {
            fold_setting_ = settings_.fold_final_level;
            fold();
            if (!identity()) {
                for (Index i = 0; i < rows_; ++i) {
                    std::fill_n(scaled_.row(i), cols_, Scalar(0));
                    detail::multiply_transposed_add(Scalar(1), metric().view(), matrix_.row(i),
                                                    scaled_.row(i));
                }
            }
            scaled_valid_ = true;
        }
        detail::multiply(matrix_.block(0, 0, rows_, cols_), reference().data(), at_reference_.data());
        // Absolute: rounding noise scales with the longest row of a level, not with what is left of a row.
        const detail::MatrixRef<const Scalar> rows = factored_rows();
        for (Index k = 0, begin = 0; k < levels_; ++k) {
            const Index end = level_ends_(k);
            Scalar longest = 0;
            for (Index i = begin; i < end; ++i) {
                longest = std::max(longest, detail::squared_norm(rows.row(i), cols_));
            }
            noise_(k) = settings_.rank_tolerance * std::sqrt(longest);
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
        if (identity()) {
            for (Index j = 0; j < cols_; ++j) {
                primal_(j) = reference()(j) + step_(j);
            }
        } else {
            detail::multiply(metric().view(), step_.data(), primal_.data());
            detail::axpy(Scalar(1), reference().data(), primal_.data(), cols_);
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
        detail::multiply(matrix_.block(begin, 0, end - begin, cols_), point_.data(), values_.data() + begin);
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
    /// level <= h (every level, in the final phase) that it would violate; that constraint and its bound are
    /// returned in `row` and `at_lower`.
    Scalar ratio_test(Index h, Index& row, bool& at_lower) noexcept {
        Scalar step = 1;
        row = -1;
        const Index end = level_ends_(std::min(h, searched_ - 1));
        detail::multiply(matrix_.block(0, 0, end, cols_), primal_.data(), values_.data());
        for (Index i = 0; i < end; ++i) {
            if (active_(i)) {
                continue;
            }
            const bool above = values_(i) - upper_(i) > settings_.feasibility_tolerance;
            if (above || lower_(i) - values_(i) > settings_.feasibility_tolerance) {
                // The iterate satisfies the bound, so the denominator is nonzero.
                const Scalar from = detail::dot(matrix_.row(i), point_.data(), cols_);
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
    bool fold_ = false;           // the last level is the choice among the optima (see `fold`)
    bool fold_identity_ = false;  // and its metric is the identity
    bool fold_setting_ = true;    // settings_.fold_final_level when fold_ was decided
    Index searched_ = 0;          // levels searched: all, or all but a folded last one
    Index phase_ = 0;             // level being solved, or levels_ for the choice among the optima
    Index changed_ = 0;           // first level whose active rows changed since the last forward solve
    Status status_ = Status::NoProblem;
    Settings<Scalar> settings_;
    Index changes_ = 0;

    Rows matrix_;  // rows of the problem
    Rows scaled_;  // rows times metric_inv_: the rows that are factored, unless the metric is the identity
    RowsVector lower_;
    RowsVector upper_;
    LevelEnds level_ends_;
    LevelsVector noise_;  // absolute rank threshold of each level
    Metric metric_inv_;   // x = metric_inv_ z
    ColsVector reference_;

    Mask active_;
    Mask lower_side_;  // active constraints held at their lower bound
    Mask locked_;      // active constraints that may no longer leave
    Mask equality_;

    Factorization lq_;
    RowsVector goal_;
    RowsVector residual_;
    RowsVector dual_;
    RowsVector values_;
    RowsVector at_reference_;  // A times the reference
    RowsVector violation_;
    RowsVector multipliers_;
    ColsVector step_;
    ColsVector primal_;  // solution of the active set
    ColsVector point_;   // iterate of the search
    ColsVector solution_;
    Metric fold_inv_;  // C of a folded last level
    ColsVector fold_reference_;
};

}  // namespace hlq
