#pragma once

// Random problem families used by the tests, benchmarks and comparisons.

#include <random>
#include <string>
#include <vector>

#include <Eigen/Dense>

namespace hlq::test {

struct Problem {
    Eigen::MatrixXd matrix;
    Eigen::VectorXd lower;
    Eigen::VectorXd upper;
    Eigen::VectorXi level_ends;

    [[nodiscard]] Eigen::Index rows() const { return matrix.rows(); }
    [[nodiscard]] Eigen::Index cols() const { return matrix.cols(); }
    [[nodiscard]] Eigen::Index levels() const { return level_ends.size(); }
    [[nodiscard]] Eigen::Index begin(Eigen::Index level) const {
        return level == 0 ? 0 : level_ends(level - 1);
    }
    [[nodiscard]] Eigen::Index size(Eigen::Index level) const { return level_ends(level) - begin(level); }
};

/// Family of random problems. Every problem ends with a level x == 0, so that its solution is unique.
enum class Family {
    /// The generator of hqp-compare's 04_random_problem: 2-8 variables, 1-6 levels of 1-5 unit rows.
    Random,
    /// Small integer rows and bounds in 3 variables, 2-4 levels, a fifth of them equalities.
    Integer,
    /// As Integer, with many rows copied (and scaled) from earlier ones.
    Degenerate,
    /// Random at robot sizes: 20-40 variables, 3-8 levels of 2-10 rows.
    Large,
};

inline const char* to_string(Family family) {
    switch (family) {
        case Family::Random:
            return "random";
        case Family::Integer:
            return "integer";
        case Family::Degenerate:
            return "degenerate";
        case Family::Large:
            return "large";
    }
    return "?";
}

namespace detail {

inline Problem assemble(const std::vector<Eigen::RowVectorXd>& rows, const std::vector<double>& lower,
                        const std::vector<double>& upper, const std::vector<int>& ends, Eigen::Index cols) {
    const auto m = static_cast<Eigen::Index>(rows.size());
    Problem p{Eigen::MatrixXd(m + cols, cols), Eigen::VectorXd(m + cols), Eigen::VectorXd(m + cols),
              Eigen::VectorXi(static_cast<Eigen::Index>(ends.size()) + 1)};
    for (Eigen::Index i = 0; i < m; ++i) {
        const auto k = static_cast<std::size_t>(i);
        p.matrix.row(i) = rows[k];
        p.lower(i) = lower[k];
        p.upper(i) = upper[k];
    }
    p.matrix.bottomRows(cols).setIdentity();
    p.lower.tail(cols).setZero();
    p.upper.tail(cols).setZero();
    for (std::size_t k = 0; k < ends.size(); ++k) {
        p.level_ends(static_cast<Eigen::Index>(k)) = ends[k];
    }
    p.level_ends(p.level_ends.size() - 1) = static_cast<int>(m + cols);
    return p;
}

inline Problem random_rows(std::mt19937& gen, int min_cols, int max_cols, int min_levels, int max_levels,
                           int min_rows, int max_rows) {
    std::uniform_int_distribution<> cols_dist(min_cols, max_cols), levels_dist(min_levels, max_levels),
        rows_dist(min_rows, max_rows);
    std::uniform_real_distribution<> value(-10.0, 10.0), bound(-20.0, 20.0), coin(0.0, 1.0);
    const int cols = cols_dist(gen);
    const int levels = levels_dist(gen);
    std::vector<int> counts(static_cast<std::size_t>(levels));
    for (auto& c : counts) {
        c = rows_dist(gen);
    }
    std::vector<Eigen::RowVectorXd> rows;
    std::vector<double> lower, upper;
    std::vector<int> ends;
    for (const int count : counts) {
        for (int r = 0; r < count; ++r) {
            Eigen::RowVectorXd a(cols);
            for (int c = 0; c < cols; ++c) {
                a(c) = value(gen);
            }
            double lo = 0, up = 0;
            if (coin(gen) < 0.3) {
                lo = up = bound(gen);
            } else {
                lo = bound(gen);
                up = bound(gen);
                if (lo > up) {
                    std::swap(lo, up);
                }
            }
            const double norm = a.norm();
            rows.emplace_back(a / norm);
            lower.push_back(lo / norm);
            upper.push_back(up / norm);
        }
        ends.push_back(static_cast<int>(rows.size()));
    }
    return assemble(rows, lower, upper, ends, cols);
}

inline Problem integer_rows(std::mt19937& gen, bool degenerate) {
    constexpr int cols = 3;
    std::uniform_int_distribution<int> coef(-2, 2), bound(-4, 4), coin(0, 9), count_dist(1, 3);
    const int levels = std::uniform_int_distribution<int>(2, 4)(gen);
    std::vector<Eigen::RowVectorXd> rows;
    std::vector<double> lower, upper;
    std::vector<int> ends;
    for (int k = 0; k < levels; ++k) {
        const int count = count_dist(gen);
        for (int r = 0; r < count; ++r) {
            Eigen::RowVectorXd a(cols);
            if (degenerate && !rows.empty() && coin(gen) < 4) {
                const auto source = std::uniform_int_distribution<std::size_t>(0, rows.size() - 1)(gen);
                a = rows[source] * double(1 + coin(gen) % 2);
            } else {
                do {
                    for (int c = 0; c < cols; ++c) {
                        a(c) = coef(gen);
                    }
                } while (a.isZero());
            }
            const int b1 = bound(gen), b2 = bound(gen);
            double lo = std::min(b1, b2), up = std::max(b1, b2);
            if (coin(gen) < 2) {
                up = lo;
            } else if (lo == up) {
                up += 1;
            }
            rows.push_back(a);
            lower.push_back(lo);
            upper.push_back(up);
        }
        ends.push_back(static_cast<int>(rows.size()));
    }
    return assemble(rows, lower, upper, ends, cols);
}

}  // namespace detail

/// Problem number `seed` of a family.
inline Problem make_problem(Family family, unsigned seed) {
    std::mt19937 gen(seed);
    switch (family) {
        case Family::Random:
            return detail::random_rows(gen, 2, 8, 1, 6, 1, 5);
        case Family::Integer:
            return detail::integer_rows(gen, false);
        case Family::Degenerate:
            return detail::integer_rows(gen, true);
        case Family::Large:
            return detail::random_rows(gen, 20, 40, 3, 8, 2, 10);
    }
    return {};
}

/// Shrinks the smallest singular value of one level (not the final x == 0) by 1e-1 to 1e-6.
inline void make_ill_conditioned(Problem& p, unsigned seed) {
    std::mt19937 gen(7000003u + seed);
    const Eigen::Index levels = p.levels() - 1;
    if (levels <= 0) {
        return;
    }
    const auto k = std::uniform_int_distribution<Eigen::Index>(0, levels - 1)(gen);
    if (p.size(k) < 2) {
        return;
    }
    auto block = p.matrix.middleRows(p.begin(k), p.size(k));
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(block, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::VectorXd sigma = svd.singularValues();
    sigma(sigma.size() - 1) *= std::pow(10.0, -std::uniform_real_distribution<double>(1, 6)(gen));
    block = svd.matrixU() * sigma.asDiagonal() * svd.matrixV().transpose();
}

/// A random symmetric positive definite metric.
inline Eigen::MatrixXd make_metric(Eigen::Index cols, unsigned seed) {
    std::mt19937 gen(1000003u + seed);
    std::uniform_real_distribution<double> value(-1.0, 1.0);
    const Eigen::MatrixXd m = Eigen::MatrixXd::NullaryExpr(cols, cols, [&] { return value(gen); });
    return m * m.transpose() + 0.1 * Eigen::MatrixXd::Identity(cols, cols);
}

}  // namespace hlq::test
