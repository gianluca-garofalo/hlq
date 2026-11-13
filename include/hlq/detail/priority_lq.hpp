#pragma once

/// @file
/// Priority-ordered LQ factorization of the active rows, with its updates and solves.

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

#include <Eigen/Dense>

#include <hlq/detail/householder.hpp>

namespace hlq::detail {

/// A compile-time capacity plus `extra`, or `Eigen::Dynamic`.
constexpr int grow(int capacity, int extra) noexcept {
    return capacity == Eigen::Dynamic ? Eigen::Dynamic : capacity + extra;
}

/**
 * Factorization A_W = K Q of the active rows of a problem, in priority order.
 *
 * Positions number the active rows in factor order: level by level, and within a level the independent rows
 * first. Row p of Q is the direction of position p, orthonormal to the others, or zero for a row that depends
 * on the rows before it. K is lower triangular, with a zero diagonal and a zero column at dependent
 * positions. The rows of a level, projected onto what the levels above leave free, are its block of K times
 * its rows of Q, so one factorization serves every level:
 *
 * - `solve` gives the hierarchical least-squares solution by forward substitution with K;
 * - `multipliers` gives the multipliers of a level's subproblem by backward substitution with K^T;
 * - `insert` and `remove` change one active row and update only the positions after it, by Givens rotations.
 *
 * Nothing above the diagonal of K is ever written, so that part stays zero. All workspace is allocated by the
 * constructor and `prepare`; no other member allocates.
 */
template <typename Scalar, int MaxRows, int MaxCols, int MaxLevels>
class PriorityLQ {
public:
    using Index = Eigen::Index;
    /// Rows of the problem (or of Q), row major.
    using RowMatrix =
        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, MaxRows, MaxCols>;
    /// Vector over rows or positions.
    using RowsVector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1, 0, MaxRows, 1>;
    /// Vector over columns.
    using ColsVector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1, 0, MaxCols, 1>;
    /// Vector over levels.
    using LevelsVector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1, 0, MaxLevels, 1>;

    PriorityLQ(Index max_rows, Index cols)
        : cols_(cols),
          row_(max_rows),
          position_(max_rows),
          level_of_(max_rows),
          coefs_(max_rows, max_rows),
          basis_(max_rows, cols),
          rows_(max_rows + 1, cols + 1),
          tail_basis_(cols + 1, cols),
          tail_lead_(max_rows + 1, max_rows),
          qr_(max_rows, max_rows),
          tau_(max_rows),
          norms_(max_rows + 1),
          entry_(max_rows + 1),
          scratch_(max_rows),
          second_(max_rows),
          coords_(max_rows),
          theta_(max_rows),
          fresh_(cols) {
        coefs_.setZero();
    }

    /// Sizes the arrays over levels (allocating only when the number of levels changes).
    void prepare(Index levels) {
        levels_ = levels;
        begin_.resize(levels + 1);
        rank_.resize(levels);
    }

    /// Number of active rows.
    [[nodiscard]] Index size() const noexcept { return size_; }
    /// Problem row at a position.
    [[nodiscard]] Index row(Index position) const noexcept { return row_(position); }
    /// Position of a problem row, or -1 when it is not active.
    [[nodiscard]] Index position(Index row) const noexcept { return position_(row); }
    /// First position of a level; `begin(levels)` is the number of active rows.
    [[nodiscard]] Index begin(Index level) const noexcept { return begin_(level); }
    /// Number of independent rows of a level.
    [[nodiscard]] Index rank(Index level) const noexcept { return rank_(level); }
    /// K, by position.
    [[nodiscard]] auto coefficients() const noexcept { return coefs_.topLeftCorner(size_, size_); }
    /// Q, by position.
    [[nodiscard]] auto basis() const noexcept { return basis_.topRows(size_); }

    /**
     * Factors the active rows from scratch by Gram-Schmidt in priority order.
     *
     * @param rows      rows of the problem, level by level
     * @param active    which rows are active
     * @param level_end end of each level in `rows`
     * @param noise     rank threshold of each level: absolute, compared with what a row has outside the rows
     *                  before it
     */
    template <typename Rows, typename Active, typename Ends>
    void factor(const Rows& rows, const Active& active, const Ends& level_end, const LevelsVector& noise) {
        size_ = 0;
        for (Index k = 0, i = 0; k < levels_; ++k) {
            begin_(k) = size_;
            for (; i < static_cast<Index>(level_end(k)); ++i) {
                level_of_(i) = k;
                position_(i) = -1;
                if (active(i)) {
                    place(size_++, i);
                }
            }
        }
        begin_(levels_) = size_;

        auto res = rows_.topLeftCorner(size_, cols_);
        for (Index p = 0; p < size_; ++p) {
            res.row(p) = rows.row(row_(p));
        }
        coefs_.topLeftCorner(size_, size_).setZero();
        entry_.head(size_) = res.rowwise().squaredNorm();

        for (Index k = 0; k < levels_; ++k) {
            const Index begin = begin_(k);
            const Index count = begin_(k + 1) - begin;
            if (count > 0) {
                // First pass: coefficients on the directions of the levels above.
                auto level = res.middleRows(begin, count);
                if (begin > 0) {
                    coefs_.block(begin, 0, count, begin).noalias() =
                        level * basis_.topRows(begin).transpose();
                    level.noalias() -= coefs_.block(begin, 0, count, begin) * basis_.topRows(begin);
                }
                auto norms = norms_.head(count);
                norms = level.rowwise().squaredNorm();
                const Scalar exact = recompute_below() * norms.maxCoeff();

                for (Index j = 0; j < count; ++j) {
                    const Index p = begin + j;
                    // Pivot on the largest residual of the level. Downdated norms lose their accuracy as the
                    // residuals vanish, so they are recomputed there.
                    Index best = 0;
                    if (norms.tail(count - j).maxCoeff(&best) < exact) {
                        norms.tail(count - j) = level.bottomRows(count - j).rowwise().squaredNorm();
                        norms.tail(count - j).maxCoeff(&best);
                    }
                    if (best > 0) {
                        level.row(j).swap(level.row(j + best));
                        coefs_.row(p).head(p).swap(coefs_.row(p + best).head(p));
                        swap_positions(p, p + best);
                        std::swap(norms(j), norms(j + best));
                        std::swap(entry_(p), entry_(p + best));
                    }

                    // A second pass keeps Q orthonormal to working precision. It is only needed when the
                    // first one cancelled most of the row (Daniel, Gragg, Kaufman and Stewart, 1976).
                    if (p > 0 && level.row(j).squaredNorm() < Scalar(0.5) * entry_(p)) {
                        scratch_.head(p).noalias() = basis_.topRows(p) * level.row(j).transpose();
                        coefs_.row(p).head(p) += scratch_.head(p).transpose();
                        level.row(j).noalias() -= scratch_.head(p).transpose() * basis_.topRows(p);
                    }

                    const Scalar norm = level.row(j).norm();
                    const Index rest = count - j - 1;
                    if (norm > noise(k)) {
                        coefs_(p, p) = norm;
                        basis_.row(p) = level.row(j) / norm;
                        if (rest > 0) {
                            coefs_.block(p + 1, p, rest, 1).noalias() =
                                level.middleRows(j + 1, rest) * basis_.row(p).transpose();
                            level.middleRows(j + 1, rest).noalias() -=
                                coefs_.block(p + 1, p, rest, 1) * basis_.row(p);
                            norms.tail(rest) -= coefs_.block(p + 1, p, rest, 1).cwiseAbs2();
                        }
                    } else {
                        basis_.row(p).setZero();
                    }
                }
            }
            count_rank(k);
        }
    }

    /**
     * Makes a problem row active and updates the positions after it.
     *
     * The row goes after the independent rows of its level. Its coefficients on every direction take two
     * passes; the positions after it keep their coefficients on the directions before it and are brought back
     * to echelon form by Givens rotations.
     */
    template <typename Rows>
    void insert(const Rows& rows, Index row, Index level, const LevelsVector& noise) {
        const Index p = begin_(level) + rank_(level);
        const Index size = size_;
        const Index tail = size - p;

        auto coef = scratch_.head(size);
        auto second = second_.head(size);
        fresh_ = rows.row(row).transpose();
        coef.noalias() = basis_.topRows(size) * fresh_;
        fresh_.noalias() -= basis_.topRows(size).transpose() * coef;
        second.noalias() = basis_.topRows(size) * fresh_;
        coef += second;
        fresh_.noalias() -= basis_.topRows(size).transpose() * second;
        const Scalar outside = fresh_.norm();

        // The positions from p on, in coordinates: the directions after p, plus the new one if any.
        Index dim = 0;
        for (Index j = p; j < size; ++j) {
            if (coefs_(j, j) > 0) {
                tail_basis_.row(dim) = basis_.row(j);
                rows_(0, dim) = coef(j);
                rows_.col(dim).segment(1, tail) = coefs_.col(j).segment(p, tail);
                ++dim;
            }
        }
        if (outside > noise(level)) {
            tail_basis_.row(dim) = fresh_.transpose() / outside;
            rows_(0, dim) = outside;
            rows_.col(dim).segment(1, tail).setZero();
            ++dim;
        }
        tail_lead_.block(1, 0, tail, p) = coefs_.block(p, 0, tail, p);
        tail_lead_.row(0).head(p) = coef.head(p).transpose();

        for (Index j = size; j > p; --j) {
            place(j, row_(j - 1));
        }
        place(p, row);
        level_of_(row) = level;
        ++size_;
        for (Index k = level + 1; k <= levels_; ++k) {
            ++begin_(k);
        }
        coefs_.block(p, 0, tail + 1, p) = tail_lead_.topLeftCorner(tail + 1, p);
        rotate_tail(p, dim, noise);
    }

    /// Makes an active problem row inactive and updates the positions after it by Givens rotations.
    void remove(Index row, const LevelsVector& noise) {
        const Index p = position_(row);
        const Index level = level_of_(row);
        const Index size = size_;
        const Index tail = size - p - 1;

        // The positions after p, in the coordinates of the directions from p on.
        Index dim = 0;
        for (Index j = p; j < size; ++j) {
            if (coefs_(j, j) > 0) {
                tail_basis_.row(dim) = basis_.row(j);
                rows_.col(dim).head(tail) = coefs_.col(j).segment(p + 1, tail);
                ++dim;
            }
        }
        tail_lead_.topLeftCorner(tail, p) = coefs_.block(p + 1, 0, tail, p);

        position_(row) = -1;
        for (Index j = p; j < size - 1; ++j) {
            place(j, row_(j + 1));
        }
        --size_;
        for (Index k = level + 1; k <= levels_; ++k) {
            --begin_(k);
        }
        coefs_.block(p, 0, tail, p) = tail_lead_.topLeftCorner(tail, p);
        rotate_tail(p, dim, noise);
    }

    /**
     * Hierarchical least-squares solution of the active rows.
     *
     * @param goal     target of each position minus its value at the reference; overwritten
     * @param residual on return, the residual (row times step minus goal) of each position
     * @param step     on return, the step from the reference, in the coordinates of the rows
     */
    void solve(RowsVector& goal, RowsVector& residual, ColsVector& step) {
        auto coords = coords_.head(size_);
        coords.setZero();
        for (Index k = 0; k < levels_; ++k) {
            const Index start = begin_(k);
            const Index dim = begin_(k + 1) - start;
            const Index rank = rank_(k);
            const Index dep = dim - rank;
            if (dim == 0) {
                continue;
            }
            auto g = goal.segment(start, dim);
            if (start > 0) {
                g.noalias() -= coefs_.block(start, 0, dim, start) * coords.head(start);
            }
            if (rank == 0) {
                residual.segment(start, dim) = -g;
                continue;
            }
            if (dep == 0) {
                coords.segment(start, rank) = g;
                coefs_.block(start, start, rank, rank)
                    .template triangularView<Eigen::Lower>()
                    .solveInPlace(coords.segment(start, rank));
                residual.segment(start, rank).setZero();
                continue;
            }

            // A level whose rows conflict: least squares on its own columns of K, F = Q [R; 0]. The normal
            // equations would square the conditioning of the level.
            auto f = qr_.block(start, start, dim, rank);
            auto tau = tau_.segment(start, rank);
            f = coefs_.block(start, start, dim, rank);
            householder_qr(f, tau, scratch_.data());
            auto rotated = residual.segment(start, dim);
            rotated = g;
            apply_qt(f, tau, rotated);
            coords.segment(start, rank) = rotated.head(rank);
            f.topLeftCorner(rank, rank)
                .template triangularView<Eigen::Upper>()
                .solveInPlace(coords.segment(start, rank));
            // Residual F y - g = Q [0; -c], with c the part of Q^T g that F cannot reach.
            rotated.head(rank).setZero();
            rotated.tail(dep) = -rotated.tail(dep);
            apply_q(f, tau, rotated);
        }
        step.noalias() = basis_.topRows(size_).transpose() * coords;
    }

    /**
     * Multipliers of the positions of levels <= h in the subproblem of level h: the residuals of level h, and
     * for each level above, the minimum-norm multipliers that balance what the levels below it leave.
     *
     * `h == levels` is the final choice among the optima, the step's squared norm: its gradient is the step,
     * whose coordinates on Q are those of the last solve, and every level is above it.
     *
     * Uses the residuals and coordinates of the last `solve`, with no change to the active rows since.
     */
    void multipliers(Index h, const RowsVector& residual, RowsVector& dual) {
        const Index start = begin_(h);
        auto theta = theta_.head(start);
        if (h == levels_) {
            theta = coords_.head(start);
        } else {
            const Index dim = begin_(h + 1) - start;
            dual.segment(start, dim) = residual.segment(start, dim);
            if (start == 0) {
                return;
            }
            theta.noalias() = coefs_.block(start, 0, dim, start).transpose() * residual.segment(start, dim);
        }

        for (Index k = h - 1; k >= 0; --k) {
            const Index first = begin_(k);
            const Index size = begin_(k + 1) - first;
            const Index rank = rank_(k);
            const Index dep = size - rank;
            if (size == 0) {
                continue;
            }
            auto lambda = dual.segment(first, size);
            if (rank == 0) {
                lambda.setZero();
                continue;
            }
            // Minimum-norm solution of F^T lambda = -theta, which lies in the range of F.
            lambda.head(rank) = -theta.segment(first, rank);
            if (dep == 0) {
                coefs_.block(first, first, rank, rank)
                    .transpose()
                    .template triangularView<Eigen::Upper>()
                    .solveInPlace(lambda.head(rank));
            } else {
                auto f = qr_.block(first, first, size, rank);
                f.topLeftCorner(rank, rank)
                    .transpose()
                    .template triangularView<Eigen::Lower>()
                    .solveInPlace(lambda.head(rank));
                lambda.tail(dep).setZero();
                apply_q(f, tau_.segment(first, rank), lambda);
            }
            if (first > 0) {
                theta.head(first).noalias() += coefs_.block(first, 0, size, first).transpose() * lambda;
            }
        }
    }

private:
    /// Downdated squared norms are recomputed below this fraction of where they started.
    static Scalar recompute_below() noexcept { return std::sqrt(std::numeric_limits<Scalar>::epsilon()); }

    void place(Index p, Index row) noexcept {
        row_(p) = row;
        position_(row) = p;
    }

    void swap_positions(Index p, Index q) noexcept {
        const Index a = row_(p);
        const Index b = row_(q);
        place(p, b);
        place(q, a);
    }

    void count_rank(Index level) noexcept {
        const Index begin = begin_(level);
        rank_(level) = (coefs_.diagonal().segment(begin, begin_(level + 1) - begin).array() > 0).count();
    }

    /**
     * Echelon form of the positions from p on, given in `rows_` as coefficients on the `dim` orthonormal rows
     * of `tail_basis_`, which are orthogonal to the directions before p.
     *
     * The coefficients are already in echelon form except for one column (a removed direction) or one row (an
     * inserted one). Each row's entries on the coordinates not yet taken are brought into the next one by
     * Givens rotations of adjacent coordinates, applied to the coefficients and to the basis: a few rotations
     * per row, each O(rows + cols).
     */
    void rotate_tail(Index p, Index dim, const LevelsVector& noise) {
        const Index tail = size_ - p;
        auto rows = rows_.topLeftCorner(tail, dim);
        auto base = tail_basis_.topRows(dim);
        auto left = norms_.head(tail);  // squared norm of each row on the coordinates not yet taken
        left = rows.rowwise().squaredNorm();
        coefs_.block(p, p, tail, tail).setZero();

        Index next = 0;  // first coordinate not yet taken by an independent row
        for (Index k = 0; k < levels_; ++k) {
            const Index end = begin_(k + 1);
            if (end < p) {
                continue;  // a level ending at p is recounted: it lost its last row when p was removed
            }
            const Index begin = std::max(p, begin_(k));
            const Scalar exact =
                end > begin ? recompute_below() * left.segment(begin - p, end - begin).maxCoeff() : Scalar(0);
            for (Index pos = begin; pos < end; ++pos) {
                const Index i = pos - p;
                // Pivot on the row of the level with the most left on the free coordinates.
                Index best = 0;
                if (left.segment(i, end - pos).maxCoeff(&best) < exact) {
                    for (Index l = i; l < end - p; ++l) {
                        left(l) = rows.row(l).tail(dim - next).squaredNorm();
                    }
                    left.segment(i, end - pos).maxCoeff(&best);
                }
                if (best > 0) {
                    rows.row(i).swap(rows.row(i + best));
                    coefs_.row(pos).head(pos).swap(coefs_.row(pos + best).head(pos));
                    swap_positions(pos, pos + best);
                    std::swap(left(i), left(i + best));
                }

                Index last = dim - 1;
                while (last > next && rows(i, last) == Scalar(0)) {
                    --last;
                }
                for (Index c = last; c > next; --c) {
                    if (rows(i, c) == Scalar(0)) {
                        continue;
                    }
                    Eigen::JacobiRotation<Scalar> rotation;
                    rotation.makeGivens(rows(i, c - 1), rows(i, c));
                    rows.bottomRows(tail - i).applyOnTheRight(c - 1, c, rotation);
                    base.applyOnTheLeft(c - 1, c, rotation.transpose());
                    rows(i, c) = Scalar(0);
                }

                if (next < dim && std::abs(rows(i, next)) > noise(k)) {
                    if (rows(i, next) < Scalar(0)) {
                        rows.col(next).tail(tail - i) *= Scalar(-1);
                        base.row(next) *= Scalar(-1);
                    }
                    // Position pos takes coordinate next: its direction, and the coefficients below on it.
                    basis_.row(pos) = base.row(next);
                    coefs_.block(pos, pos, tail - i, 1) = rows.col(next).tail(tail - i);
                    left.tail(tail - i - 1) -= rows.col(next).tail(tail - i - 1).cwiseAbs2();
                    ++next;
                } else {
                    // Dependent: what is left of it is below the rank threshold.
                    basis_.row(pos).setZero();
                    if (next < dim) {
                        rows.row(i).tail(dim - next).setZero();
                    }
                }
            }
            count_rank(k);
        }
    }

    using Positions = Eigen::Matrix<Index, Eigen::Dynamic, 1, 0, MaxRows, 1>;
    using LevelIndices = Eigen::Matrix<Index, Eigen::Dynamic, 1, 0, MaxLevels, 1>;
    using LevelStarts = Eigen::Matrix<Index, Eigen::Dynamic, 1, 0, grow(MaxLevels, 1), 1>;
    using Square = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, MaxRows, MaxRows>;
    using ColumnSquare =
        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor, MaxRows, MaxRows>;
    using Tail = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, grow(MaxRows, 1),
                               grow(MaxCols, 1)>;
    using TailBasis =
        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, grow(MaxCols, 1), MaxCols>;
    using TailLead =
        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, grow(MaxRows, 1), MaxRows>;
    using TailVector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1, 0, grow(MaxRows, 1), 1>;

    Index cols_;
    Index levels_ = 0;
    Index size_ = 0;
    Positions row_;         // problem row of each position
    Positions position_;    // position of each problem row, or -1
    Positions level_of_;    // level of each problem row
    LevelStarts begin_;     // first position of each level, then the number of positions
    LevelIndices rank_;     // independent rows of each level
    Square coefs_;          // K
    RowMatrix basis_;       // Q
    Tail rows_;             // rows being factored or rotated
    TailBasis tail_basis_;  // orthonormal coordinates of the positions after a change
    TailLead tail_lead_;    // their coefficients on the directions before it, while positions shift
    ColumnSquare qr_;       // Householder QR of each conflicting level, in its diagonal block
    RowsVector tau_;        // and its coefficients
    TailVector norms_;
    TailVector entry_;
    RowsVector scratch_;
    RowsVector second_;
    RowsVector coords_;
    RowsVector theta_;
    ColsVector fresh_;
};

}  // namespace hlq::detail
