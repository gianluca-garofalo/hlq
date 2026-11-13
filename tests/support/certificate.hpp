#pragma once

// An optimality certificate for hierarchical least squares, independent of the solver.
//
// x solves level k when, with the levels above held at the violations they have at x, it minimizes the
// squared violation of level k. The problem is convex, so this holds exactly when the gradient of level k at
// x is balanced by multipliers of the bounds of the levels above that are tight at x, with the sign that
// their side allows. Finding the multipliers is a non-negative least-squares problem, solved here by Lawson
// and Hanson's method. By induction over the levels, x is then a lexicographic optimum.

#include <algorithm>
#include <cmath>
#include <vector>

#include <Eigen/Dense>

#include "problems.hpp"

namespace hlq::test {

/// min ||M v - b|| over v >= 0 (Lawson and Hanson, 1974). Returns the residual norm.
inline double nonnegative_least_squares(const Eigen::MatrixXd& m, const Eigen::VectorXd& b) {
    const Eigen::Index n = m.cols();
    Eigen::VectorXd v = Eigen::VectorXd::Zero(n);
    std::vector<bool> passive(static_cast<std::size_t>(n), false);
    const double tolerance = 1e-12 * (1 + m.norm() * b.norm());
    for (Eigen::Index iteration = 0; iteration < 3 * n + 10; ++iteration) {
        const Eigen::VectorXd w = m.transpose() * (b - m * v);
        Eigen::Index best = -1;
        for (Eigen::Index j = 0; j < n; ++j) {
            if (!passive[static_cast<std::size_t>(j)] && w(j) > tolerance && (best < 0 || w(j) > w(best))) {
                best = j;
            }
        }
        if (best < 0) {
            break;
        }
        passive[static_cast<std::size_t>(best)] = true;
        for (Eigen::Index inner = 0; inner <= n; ++inner) {
            std::vector<Eigen::Index> set;
            for (Eigen::Index j = 0; j < n; ++j) {
                if (passive[static_cast<std::size_t>(j)]) {
                    set.push_back(j);
                }
            }
            Eigen::MatrixXd mp(m.rows(), static_cast<Eigen::Index>(set.size()));
            for (std::size_t c = 0; c < set.size(); ++c) {
                mp.col(static_cast<Eigen::Index>(c)) = m.col(set[c]);
            }
            const Eigen::VectorXd sp = mp.colPivHouseholderQr().solve(b);
            Eigen::VectorXd s = Eigen::VectorXd::Zero(n);
            for (std::size_t c = 0; c < set.size(); ++c) {
                s(set[c]) = sp(static_cast<Eigen::Index>(c));
            }
            bool positive = true;
            double alpha = 1;
            for (const Eigen::Index j : set) {
                if (s(j) <= 0) {
                    positive = false;
                    alpha = std::min(alpha, v(j) / (v(j) - s(j)));
                }
            }
            if (positive) {
                v = s;
                break;
            }
            v += alpha * (s - v);
            for (const Eigen::Index j : set) {
                if (v(j) <= 1e-15) {
                    v(j) = 0;
                    passive[static_cast<std::size_t>(j)] = false;
                }
            }
        }
    }
    return (m * v - b).norm();
}

struct Certificate {
    bool optimal = true;
    /// First level that fails, or the number of levels for the final tie-break (metric and reference).
    Eigen::Index level = -1;
    /// Unbalanced part of its gradient, relative to the gradient's norm plus one.
    double residual = 0;
};

/**
 * Checks that x is the lexicographic optimum of p, and the point of the optimal set closest to `reference` in
 * `metric` (pass an empty metric for the identity).
 *
 * @param tight    a bound counts as tight within this (absolute)
 * @param balanced a gradient counts as balanced within this (relative to its norm plus one)
 */
inline Certificate certify(const Problem& p, const Eigen::VectorXd& x, const Eigen::MatrixXd& metric = {},
                           const Eigen::VectorXd& reference = {}, double tight = 1e-7,
                           double balanced = 1e-6) {
    const Eigen::VectorXd values = p.matrix * x;
    const Eigen::VectorXd violation = values - values.cwiseMax(p.lower).cwiseMin(p.upper);

    // Columns of the multipliers of the bounds tight at x among rows [0, end), with the sign their side
    // allows.
    auto tight_columns = [&](Eigen::Index end) {
        std::vector<Eigen::VectorXd> columns;
        for (Eigen::Index i = 0; i < end; ++i) {
            const Eigen::VectorXd a = p.matrix.row(i).transpose();
            // A violated bound stays at its violation; a bound within `tight` of the value is tight (both,
            // for an equality).
            const bool upper =
                violation(i) > tight || (violation(i) >= -tight && values(i) >= p.upper(i) - tight);
            const bool lower =
                violation(i) < -tight || (violation(i) <= tight && values(i) <= p.lower(i) + tight);
            if (upper) {
                columns.push_back(a);
            }
            if (lower) {
                columns.push_back(-a);
            }
        }
        Eigen::MatrixXd m(p.cols(), static_cast<Eigen::Index>(columns.size()));
        for (std::size_t c = 0; c < columns.size(); ++c) {
            m.col(static_cast<Eigen::Index>(c)) = columns[c];
        }
        return m;
    };

    auto check = [&](const Eigen::VectorXd& gradient, Eigen::Index end, Eigen::Index level) {
        const Eigen::MatrixXd m = tight_columns(end);
        const double unbalanced = m.cols() == 0 ? gradient.norm() : nonnegative_least_squares(m, -gradient);
        const double relative = unbalanced / (1 + gradient.norm());
        return Certificate{relative <= balanced, level, relative};
    };

    for (Eigen::Index k = 0; k < p.levels(); ++k) {
        const Eigen::Index begin = p.begin(k);
        const Eigen::VectorXd gradient =
            p.matrix.middleRows(begin, p.size(k)).transpose() * violation.segment(begin, p.size(k));
        if (Certificate c = check(gradient, begin, k); !c.optimal) {
            return c;
        }
    }
    const Eigen::VectorXd offset = reference.size() > 0 ? Eigen::VectorXd(x - reference) : x;
    const Eigen::VectorXd gradient = metric.size() > 0 ? Eigen::VectorXd(metric * offset) : offset;
    return check(gradient, p.rows(), p.levels());
}

}  // namespace hlq::test
