#pragma once

/// @file
/// Priority-ordered LQ factorization of the active rows, with its updates and solves.

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

#include <hlq/detail/dense.hpp>
#include <hlq/types.hpp>

namespace hlq::detail {

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
    /// Rows of the problem, row major.
    using Rows = MatrixRef<const Scalar>;
    /// Vector over rows or positions.
    using RowsVector = Vector<Scalar, MaxRows>;
    /// Vector over columns.
    using ColsVector = Vector<Scalar, MaxCols>;
    /// Vector over levels.
    using LevelsVector = Vector<Scalar, MaxLevels>;

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
        coefs_.fill(Scalar(0));
    }

    /// Sizes the arrays over levels (allocating only when there are more levels than before).
    void prepare(Index levels) {
        levels_ = levels;
        begin_.resize(levels + 1);
        rank_.resize(levels);
        level_step_.resize(levels, cols_);
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
    [[nodiscard]] MatrixRef<const Scalar> coefficients() const noexcept {
        return coefs_.block(0, 0, size_, size_);
    }
    /// Q, by position.
    [[nodiscard]] MatrixRef<const Scalar> basis() const noexcept { return basis_.block(0, 0, size_, cols_); }
    /// Number of levels in the factorization: the first ones; the active rows of the others are left out.
    [[nodiscard]] Index factored() const noexcept { return factored_; }

    /**
     * Factors the active rows from scratch by Gram-Schmidt in priority order.
     *
     * @param rows      rows of the problem, level by level
     * @param active    which rows are active
     * @param level_end end of each level in `rows`
     * @param noise     rank threshold of each level: absolute, compared with what a row has outside the rows
     *                  before it
     * @param count     number of levels to factor, the first ones (all by default); `append` adds the others
     */
    template <typename Active, typename Ends>
    void factor(Rows rows, const Active& active, const Ends& level_end, const LevelsVector& noise,
                Index count = -1) {
        size_ = 0;
        factored_ = 0;
        for (Index k = 0, i = 0; k < levels_; ++k) {
            begin_(k) = size_;
            for (; i < static_cast<Index>(level_end(k)); ++i) {
                level_of_(i) = k;
                position_(i) = -1;
            }
        }
        begin_(levels_) = 0;
        count = count < 0 ? levels_ : count;
        for (Index k = 0, i = 0; k < count; ++k) {
            append(rows, active, i, static_cast<Index>(level_end(k)), noise);
            i = static_cast<Index>(level_end(k));
        }
    }

    /// Leaves the levels after `level` out of the factorization, at no cost; `append` brings them back.
    void truncate(Index level) noexcept {
        const Index end = begin_(level + 1);
        for (Index p = end; p < size_; ++p) {
            position_(row_(p)) = -1;
        }
        size_ = end;
        for (Index k = level + 1; k < levels_; ++k) {
            begin_(k + 1) = end;
            rank_(k) = 0;
        }
        factored_ = level + 1;
    }

    /**
     * Factors the active rows of the next level, problem rows `[first, last)`, after the levels in the
     * factorization, by Gram-Schmidt with pivoting.
     */
    template <typename Active>
    void append(Rows rows, const Active& active, Index first, Index last, const LevelsVector& noise) {
        const Index k = factored_++;
        const Index begin = size_;
        begin_(k) = begin;
        for (Index i = first; i < last; ++i) {
            if (active(i)) {
                place(size_++, i);
            }
        }
        for (Index j = k + 1; j <= levels_; ++j) {
            begin_(j) = size_;
        }
        const Index count = size_ - begin;
        const MatrixRef<Scalar> level = rows_.block(begin, 0, count, cols_);
        for (Index j = 0; j < count; ++j) {
            std::copy_n(rows.row(row_(begin + j)), cols_, level.row(j));
            std::fill_n(coefs_.row(begin + j), size_, Scalar(0));
            entry_(begin + j) = squared_norm(level.row(j), cols_);
        }
        if (count > 0) {
            // First pass: coefficients on the directions of the levels above.
            for (Index j = 0; j < count && begin > 0; ++j) {
                Scalar* coef = coefs_.row(begin + j);
                multiply(basis_.block(0, 0, begin, cols_), level.row(j), coef);
                multiply_transposed_add(Scalar(-1), basis_.block(0, 0, begin, cols_), coef, level.row(j));
            }
            Scalar* norms = norms_.data();
            for (Index j = 0; j < count; ++j) {
                norms[j] = squared_norm(level.row(j), cols_);
            }
            const Scalar exact = recompute_below() * max_of(norms, count);

            for (Index j = 0; j < count; ++j) {
                const Index p = begin + j;
                // Pivot on the largest residual of the level. Downdated norms lose their accuracy as the
                // residuals vanish, so they are recomputed there.
                Index best = argmax(norms + j, count - j);
                if (norms[j + best] < exact) {
                    for (Index l = j; l < count; ++l) {
                        norms[l] = squared_norm(level.row(l), cols_);
                    }
                    best = argmax(norms + j, count - j);
                }
                if (best > 0) {
                    std::swap_ranges(level.row(j), level.row(j) + cols_, level.row(j + best));
                    std::swap_ranges(coefs_.row(p), coefs_.row(p) + p, coefs_.row(p + best));
                    swap_positions(p, p + best);
                    std::swap(norms[j], norms[j + best]);
                    std::swap(entry_(p), entry_(p + best));
                }

                // A second pass keeps Q orthonormal to working precision. It is only needed when the
                // first one cancelled most of the row (Daniel, Gragg, Kaufman and Stewart, 1976), and not
                // for a row left below the rank threshold, which gets no direction.
                const Scalar once = squared_norm(level.row(j), cols_);
                if (p > 0 && once < Scalar(0.5) * entry_(p) && once > noise(k) * noise(k)) {
                    Scalar* again = scratch_.data();
                    multiply(basis_.block(0, 0, p, cols_), level.row(j), again);
                    axpy(Scalar(1), again, coefs_.row(p), p);
                    multiply_transposed_add(Scalar(-1), basis_.block(0, 0, p, cols_), again, level.row(j));
                }

                const Scalar norm = std::sqrt(squared_norm(level.row(j), cols_));
                if (norm > noise(k)) {
                    coefs_(p, p) = norm;
                    Scalar* direction = basis_.row(p);
                    for (Index c = 0; c < cols_; ++c) {
                        direction[c] = level(j, c) / norm;
                    }
                    for (Index l = j + 1; l < count; ++l) {
                        const Scalar on = dot(level.row(l), direction, cols_);
                        coefs_(begin + l, p) = on;
                        axpy(-on, direction, level.row(l), cols_);
                        norms[l] -= on * on;
                    }
                } else {
                    std::fill_n(basis_.row(p), cols_, Scalar(0));
                }
            }
        }
        count_rank(k);
    }

    /**
     * Makes a problem row active and updates the positions after it.
     *
     * The row goes after the independent rows of its level. Its coefficients on every direction take one
     * pass, or two when the first cancels most of the row; the positions after it keep their coefficients on
     * the directions before it and are brought back to echelon form by Givens rotations.
     */
    void insert(Rows rows, Index row, Index level, const LevelsVector& noise) {
        const Index p = begin_(level) + rank_(level);
        const Index size = size_;
        const Index tail = size - p;
        const MatrixRef<const Scalar> basis = basis_.block(0, 0, size, cols_);

        Scalar* coef = scratch_.data();
        Scalar* fresh = fresh_.data();
        std::copy_n(rows.row(row), cols_, fresh);
        const Scalar length = squared_norm(fresh, cols_);
        multiply(basis, fresh, coef);
        multiply_transposed_add(Scalar(-1), basis, coef, fresh);
        const Scalar once = squared_norm(fresh, cols_);
        if (once < Scalar(0.5) * length && once > noise(level) * noise(level)) {
            // The first pass cancelled most of the row: a second one restores orthogonality.
            Scalar* second = second_.data();
            multiply(basis, fresh, second);
            axpy(Scalar(1), second, coef, size);
            multiply_transposed_add(Scalar(-1), basis, second, fresh);
        }
        const Scalar outside = std::sqrt(squared_norm(fresh, cols_));

        // The positions from p on, in coordinates: the directions after p, plus the new one if any.
        Index dim = 0;
        for (Index j = p; j < size; ++j) {
            if (coefs_(j, j) > 0) {
                std::copy_n(basis_.row(j), cols_, tail_basis_.row(dim));
                rows_(0, dim) = coef[j];
                for (Index t = 0; t < tail; ++t) {
                    rows_(1 + t, dim) = coefs_(p + t, j);
                }
                ++dim;
            }
        }
        if (outside > noise(level)) {
            Scalar* direction = tail_basis_.row(dim);
            for (Index c = 0; c < cols_; ++c) {
                direction[c] = fresh[c] / outside;
            }
            rows_(0, dim) = outside;
            for (Index t = 0; t < tail; ++t) {
                rows_(1 + t, dim) = Scalar(0);
            }
            ++dim;
        }
        for (Index t = 0; t < tail; ++t) {
            std::copy_n(coefs_.row(p + t), p, tail_lead_.row(1 + t));
        }
        std::copy_n(coef, p, tail_lead_.row(0));

        for (Index j = size; j > p; --j) {
            place(j, row_(j - 1));
        }
        place(p, row);
        level_of_(row) = level;
        ++size_;
        for (Index k = level + 1; k <= levels_; ++k) {
            ++begin_(k);
        }
        for (Index t = 0; t <= tail; ++t) {
            std::copy_n(tail_lead_.row(t), p, coefs_.row(p + t));
        }
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
                std::copy_n(basis_.row(j), cols_, tail_basis_.row(dim));
                for (Index t = 0; t < tail; ++t) {
                    rows_(t, dim) = coefs_(p + 1 + t, j);
                }
                ++dim;
            }
        }
        for (Index t = 0; t < tail; ++t) {
            std::copy_n(coefs_.row(p + 1 + t), p, tail_lead_.row(t));
        }

        position_(row) = -1;
        for (Index j = p; j < size - 1; ++j) {
            place(j, row_(j + 1));
        }
        --size_;
        for (Index k = level + 1; k <= levels_; ++k) {
            --begin_(k);
        }
        for (Index t = 0; t < tail; ++t) {
            std::copy_n(tail_lead_.row(t), p, coefs_.row(p + t));
        }
        rotate_tail(p, dim, noise);
    }

    /**
     * Hierarchical least-squares solution of the active rows.
     *
     * The levels before `from` are taken from the last solve: neither their positions nor their goals may
     * have changed since.
     *
     * @param goal     target of each position minus its value at the reference; overwritten
     * @param residual on return, the residual (row times step minus goal) of each position
     * @param step     on return, the step from the reference, in the coordinates of the rows
     * @param from     first level that changed since the last solve
     */
    void solve(RowsVector& goal, RowsVector& residual, ColsVector& step, Index from = 0) {
        Scalar* coords = coords_.data();
        for (Index k = from; k < levels_; ++k) {
            const Index start = begin_(k);
            const Index dim = begin_(k + 1) - start;
            const Index rank = rank_(k);
            const Index dep = dim - rank;
            Scalar* part = level_step_.row(k);
            std::fill_n(part, cols_, Scalar(0));
            if (dim == 0) {
                continue;
            }
            Scalar* g = goal.data() + start;
            multiply<true>(coefs_.block(start, 0, dim, start), coords, g, Scalar(-1));
            if (rank == 0) {
                std::fill_n(coords + start, dim, Scalar(0));
                for (Index r = 0; r < dim; ++r) {
                    residual(start + r) = -g[r];
                }
                continue;
            }
            if (dep == 0) {
                std::copy_n(g, rank, coords + start);
                solve_lower(coefs_.block(start, start, rank, rank), coords + start);
                std::fill_n(residual.data() + start, rank, Scalar(0));
            } else {
                // A level whose rows conflict: least squares on its own columns of K, F = Q [R; 0]. The
                // normal equations would square the conditioning of the level. The QR holds F^T.
                const MatrixRef<Scalar> ft = qr_.block(start, start, rank, dim);
                for (Index c = 0; c < rank; ++c) {
                    for (Index r = 0; r < dim; ++r) {
                        ft(c, r) = coefs_(start + r, start + c);
                    }
                }
                Scalar* tau = tau_.data() + start;
                householder_qr(ft, tau);
                Scalar* rotated = residual.data() + start;
                std::copy_n(g, dim, rotated);
                apply_qt(ft, tau, rotated);
                std::copy_n(rotated, rank, coords + start);
                solve_lower_transposed(ft.block(0, 0, rank, rank), coords + start);
                // Residual F y - g = Q [0; -c], with c the part of Q^T g that F cannot reach.
                std::fill_n(rotated, rank, Scalar(0));
                scale(Scalar(-1), rotated + rank, dep);
                apply_q(ft, tau, rotated);
                // The dependent positions have zero rows of Q, so their coordinates do not matter.
                std::fill_n(coords + start + rank, dep, Scalar(0));
            }
            multiply_transposed_add(Scalar(1), basis_.block(start, 0, rank, cols_), coords + start, part);
        }
        std::fill_n(step.data(), cols_, Scalar(0));
        for (Index k = 0; k < levels_; ++k) {
            axpy(Scalar(1), level_step_.row(k), step.data(), cols_);
        }
    }

    /**
     * Multipliers of the positions of levels <= h in the subproblem of level h: the residuals of level h, and
     * for each level above, the minimum-norm multipliers that balance what the levels below it leave.
     *
     * `h == factored()` is the final choice among the optima, the step's squared norm: its gradient is the
     * step, whose coordinates on Q are those of the last solve, and every level in the factorization is above
     * it.
     *
     * Uses the residuals and coordinates of the last `solve`, with no change to the active rows since.
     */
    void multipliers(Index h, const RowsVector& residual, RowsVector& dual) {
        const Index start = begin_(h);
        Scalar* theta = theta_.data();
        if (h >= factored_) {
            std::copy_n(coords_.data(), start, theta);
        } else {
            const Index dim = begin_(h + 1) - start;
            std::copy_n(residual.data() + start, dim, dual.data() + start);
            if (start == 0) {
                return;
            }
            std::fill_n(theta, start, Scalar(0));
            multiply_transposed_add(Scalar(1), coefs_.block(start, 0, dim, start), residual.data() + start,
                                    theta);
        }
        backward(std::min(h, factored_), theta, dual);
    }

    /**
     * Multipliers of every position for an objective of the given gradient (in the coordinates of the rows)
     * above all the levels in the factorization: the minimum-norm ones within each level, as `multipliers`.
     * Uses the QRs of the last `solve`.
     */
    void multipliers_of(const Scalar* gradient, RowsVector& dual) {
        Scalar* theta = theta_.data();
        multiply(basis_.block(0, 0, size_, cols_), gradient, theta);
        backward(factored_, theta, dual);
    }

private:
    /// Backward substitution through the levels before `above`, from theta, the gradient's coordinates.
    void backward(Index above, Scalar* theta, RowsVector& dual) {
        for (Index k = above - 1; k >= 0; --k) {
            const Index first = begin_(k);
            const Index size = begin_(k + 1) - first;
            const Index rank = rank_(k);
            const Index dep = size - rank;
            if (size == 0) {
                continue;
            }
            Scalar* lambda = dual.data() + first;
            if (rank == 0) {
                std::fill_n(lambda, size, Scalar(0));
                continue;
            }
            // Minimum-norm solution of F^T lambda = -theta, which lies in the range of F.
            for (Index r = 0; r < rank; ++r) {
                lambda[r] = -theta[first + r];
            }
            if (dep == 0) {
                solve_lower_transposed(coefs_.block(first, first, rank, rank), lambda);
            } else {
                const MatrixRef<const Scalar> ft = qr_.block(first, first, rank, size);
                solve_lower(ft.block(0, 0, rank, rank), lambda);
                std::fill_n(lambda + rank, dep, Scalar(0));
                apply_q(ft, tau_.data() + first, lambda);
            }
            if (first > 0) {
                multiply_transposed_add(Scalar(1), coefs_.block(first, 0, size, first), lambda, theta);
            }
        }
    }

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
        Index rank = 0;
        for (Index p = begin_(level); p < begin_(level + 1); ++p) {
            rank += coefs_(p, p) > 0 ? 1 : 0;
        }
        rank_(level) = rank;
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
        const MatrixRef<Scalar> rows = rows_.block(0, 0, tail, dim);
        const MatrixRef<Scalar> base = tail_basis_.block(0, 0, dim, cols_);
        Scalar* left = norms_.data();  // squared norm of each row on the coordinates not yet taken
        for (Index i = 0; i < tail; ++i) {
            left[i] = squared_norm(rows.row(i), dim);
            std::fill_n(coefs_.row(p + i) + p, tail, Scalar(0));
        }

        Index next = 0;  // first coordinate not yet taken by an independent row
        for (Index k = 0; k < levels_; ++k) {
            const Index end = begin_(k + 1);
            if (end < p) {
                continue;  // a level ending at p is recounted: it lost its last row when p was removed
            }
            const Index begin = std::max(p, begin_(k));
            const Scalar exact =
                end > begin ? recompute_below() * max_of(left + begin - p, end - begin) : Scalar(0);
            for (Index pos = begin; pos < end; ++pos) {
                const Index i = pos - p;
                // Pivot on the row of the level with the most left on the free coordinates.
                Index best = argmax(left + i, end - pos);
                if (left[i + best] < exact) {
                    for (Index l = i; l < end - p; ++l) {
                        left[l] = squared_norm(rows.row(l) + next, dim - next);
                    }
                    best = argmax(left + i, end - pos);
                }
                if (best > 0) {
                    std::swap_ranges(rows.row(i), rows.row(i) + dim, rows.row(i + best));
                    std::swap_ranges(coefs_.row(pos), coefs_.row(pos) + pos, coefs_.row(pos + best));
                    swap_positions(pos, pos + best);
                    std::swap(left[i], left[i + best]);
                }

                Index last = dim - 1;
                while (last > next && rows(i, last) == Scalar(0)) {
                    --last;
                }
                for (Index c = last; c > next; --c) {
                    if (rows(i, c) == Scalar(0)) {
                        continue;
                    }
                    const Rotation<Scalar> g = make_rotation(rows(i, c - 1), rows(i, c));
                    rotate_strided(g, &rows(i, c - 1), &rows(i, c), tail - i, rows.stride);
                    rotate(g, base.row(c - 1), base.row(c), cols_);
                    rows(i, c) = Scalar(0);
                }

                if (next < dim && std::abs(rows(i, next)) > noise(k)) {
                    if (rows(i, next) < Scalar(0)) {
                        for (Index l = i; l < tail; ++l) {
                            rows(l, next) = -rows(l, next);
                        }
                        scale(Scalar(-1), base.row(next), cols_);
                    }
                    // Position pos takes coordinate next: its direction, and the coefficients below on it.
                    std::copy_n(base.row(next), cols_, basis_.row(pos));
                    for (Index l = i; l < tail; ++l) {
                        coefs_(p + l, pos) = rows(l, next);
                    }
                    for (Index l = i + 1; l < tail; ++l) {
                        left[l] -= rows(l, next) * rows(l, next);
                    }
                    ++next;
                } else {
                    // Dependent: what is left of it is below the rank threshold.
                    std::fill_n(basis_.row(pos), cols_, Scalar(0));
                    if (next < dim) {
                        std::fill_n(rows.row(i) + next, dim - next, Scalar(0));
                    }
                }
            }
            count_rank(k);
        }
    }

    using Positions = Vector<Index, MaxRows>;
    using LevelIndices = Vector<Index, MaxLevels>;
    using LevelStarts = Vector<Index, grow(MaxLevels, 1)>;
    using Square = Matrix<Scalar, MaxRows, MaxRows>;
    using TailVector = Vector<Scalar, grow(MaxRows, 1)>;

    Index cols_;
    Index levels_ = 0;
    Index factored_ = 0;  // levels in the factorization
    Index size_ = 0;
    Positions row_;                           // problem row of each position
    Positions position_;                      // position of each problem row, or -1
    Positions level_of_;                      // level of each problem row
    LevelStarts begin_;                       // first position of each level, then the number of positions
    LevelIndices rank_;                       // independent rows of each level
    Square coefs_;                            // K
    Matrix<Scalar, MaxRows, MaxCols> basis_;  // Q
    Matrix<Scalar, grow(MaxRows, 1), grow(MaxCols, 1)> rows_;  // rows being factored or rotated
    Matrix<Scalar, grow(MaxCols, 1), MaxCols> tail_basis_;     // coordinates of the positions after a change
    Matrix<Scalar, grow(MaxRows, 1), MaxRows> tail_lead_;  // their coefficients before it, while they shift
    Square qr_;       // Householder QR of each conflicting level, transposed, in its diagonal block
    RowsVector tau_;  // and its coefficients
    TailVector norms_;
    TailVector entry_;
    RowsVector scratch_;
    RowsVector second_;
    RowsVector coords_;
    RowsVector theta_;
    ColsVector fresh_;
    Matrix<Scalar, MaxLevels, MaxCols> level_step_;  // each level's part of the step of the last solve
};

}  // namespace hlq::detail
