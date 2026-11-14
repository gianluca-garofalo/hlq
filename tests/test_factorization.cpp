// The priority-ordered LQ factorization against an SVD reference, from scratch and after updates.

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include <hlq/detail/priority_lq.hpp>

#include "support/reference.hpp"

namespace {

using Factor = hlq::detail::PriorityLQ<double, hlq::dynamic, hlq::dynamic, hlq::dynamic>;
using Index = hlq::Index;
using RowMajor = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

/// A view of the factorization as an Eigen matrix.
Eigen::MatrixXd to_eigen(hlq::detail::MatrixRef<const double> m) {
    return Eigen::Map<const RowMajor, 0, Eigen::OuterStride<>>(m.data, m.rows, m.cols,
                                                               Eigen::OuterStride<>(m.stride));
}

/// Random equality hierarchy with rank-deficient levels, rows that depend on the levels above, and an
/// optional ill-conditioned level.
struct Hierarchy {
    Eigen::MatrixXd rows;
    RowMajor row_major;
    Eigen::VectorXd targets;
    Eigen::VectorXi ends;
    std::vector<Eigen::MatrixXd> level_rows;
    std::vector<Eigen::VectorXd> level_targets;

    Hierarchy(unsigned seed, bool ill = false) {
        std::mt19937 gen(seed);
        std::normal_distribution<double> normal;
        auto gaussian = [&](Index r, Index c) {
            return Eigen::MatrixXd::NullaryExpr(r, c, [&] { return normal(gen); });
        };
        const auto n = std::uniform_int_distribution<Index>(2, 11)(gen);
        const auto levels = std::uniform_int_distribution<Index>(1, 5)(gen);
        std::uniform_real_distribution<double> coin(0, 1);
        for (Index k = 0; k < levels; ++k) {
            const auto m = std::uniform_int_distribution<Index>(1, n + 2)(gen);
            Eigen::MatrixXd a = gaussian(m, n);
            const double r = coin(gen);
            if (r < 0.3 && m > 1) {  // rank deficient
                const auto q = std::uniform_int_distribution<Index>(1, m - 1)(gen);
                a = gaussian(m, q) * gaussian(q, n);
            }
            if (r > 0.7 && !level_rows.empty()) {  // a row in the span of the levels above
                Eigen::MatrixXd above(0, n);
                for (const auto& b : level_rows) {
                    Eigen::MatrixXd stacked(above.rows() + b.rows(), n);
                    stacked << above, b;
                    above = stacked;
                }
                a.row(0) = gaussian(1, above.rows()) * above;
            }
            if (ill && m > 1) {
                Eigen::JacobiSVD<Eigen::MatrixXd> svd(a, Eigen::ComputeThinU | Eigen::ComputeThinV);
                Eigen::VectorXd s = svd.singularValues();
                s(s.size() - 1) *= std::pow(10.0, -std::uniform_real_distribution<double>(1, 6)(gen));
                a = svd.matrixU() * s.asDiagonal() * svd.matrixV().transpose();
            }
            level_rows.push_back(a);
            level_targets.push_back(gaussian(m, 1));
        }
        Index total = 0;
        for (const auto& b : level_rows) {
            total += b.rows();
        }
        rows.resize(total, n);
        targets.resize(total);
        ends.resize(levels);
        for (Index k = 0, at = 0; k < levels; ++k) {
            const auto& b = level_rows[static_cast<std::size_t>(k)];
            rows.middleRows(at, b.rows()) = b;
            targets.segment(at, b.rows()) = level_targets[static_cast<std::size_t>(k)];
            at += b.rows();
            ends(k) = static_cast<int>(at);
        }
        row_major = rows;
    }

    /// The rows as the factorization reads them.
    [[nodiscard]] Factor::Rows view() const {
        return {row_major.data(), row_major.rows(), row_major.cols(), row_major.cols()};
    }

    [[nodiscard]] Factor::LevelsVector noise(double tolerance = 1e-9) const {
        Factor::LevelsVector noise(ends.size());
        for (Index k = 0, begin = 0; k < ends.size(); ++k) {
            noise(k) = tolerance * rows.middleRows(begin, ends(k) - begin).rowwise().norm().maxCoeff();
            begin = ends(k);
        }
        return noise;
    }
};

/// K Q reproduces the active rows, Q is orthonormal (zero at dependent positions) and K lower triangular.
void expect_consistent(const Factor& f, const Eigen::MatrixXd& rows) {
    const Eigen::MatrixXd k = to_eigen(f.coefficients());
    const Eigen::MatrixXd q = to_eigen(f.basis());
    Eigen::MatrixXd active(f.size(), rows.cols());
    for (Index p = 0; p < f.size(); ++p) {
        active.row(p) = rows.row(f.row(p));
    }
    EXPECT_LT((k * q - active).norm(), 1e-10 * (1 + active.norm()));
    for (Index p = 0; p < f.size(); ++p) {
        if (k(p, p) > 0) {
            EXPECT_NEAR(q.row(p).norm(), 1.0, 1e-12);
        } else {
            EXPECT_EQ(q.row(p).norm(), 0.0);
        }
    }
    EXPECT_LT((q * q.transpose()
               - Eigen::MatrixXd(k.diagonal()
                                     .array()
                                     .cast<double>()
                                     .unaryExpr([](double d) { return d > 0 ? 1.0 : 0.0; })
                                     .matrix()
                                     .asDiagonal()))
                  .norm(),
              1e-12 * static_cast<double>(1 + f.size()));
    EXPECT_EQ(Eigen::MatrixXd(k.triangularView<Eigen::StrictlyUpper>()).norm(), 0.0);
}

Eigen::VectorXd solve(Factor& f, const Hierarchy& h) {
    Factor::RowsVector goal(f.size()), residual(f.size());
    Factor::ColsVector step(h.rows.cols());
    for (Index p = 0; p < f.size(); ++p) {
        goal(p) = h.targets(f.row(p));
    }
    f.solve(goal, residual, step);
    return Eigen::Map<const Eigen::VectorXd>(step.data(), step.size());
}

}  // namespace

TEST(PriorityLQ, FactorsAllRows) {
    for (unsigned seed = 0; seed < 500; ++seed) {
        const Hierarchy h(seed);
        Factor f(h.rows.rows(), h.rows.cols());
        f.prepare(h.ends.size());
        f.factor(h.view(), Eigen::Array<bool, Eigen::Dynamic, 1>::Ones(h.rows.rows()), h.ends, h.noise());
        expect_consistent(f, h.rows);
        const auto reference = hlq::test::reference_solve(h.level_rows, h.level_targets);
        for (Index k = 0; k < h.ends.size(); ++k) {
            EXPECT_EQ(f.rank(k), reference.levels[static_cast<std::size_t>(k)].rank) << "seed " << seed;
        }
    }
}

TEST(PriorityLQ, SolvesLikeTheReference) {
    for (const bool ill : {false, true}) {
        int differ = 0;
        for (unsigned seed = 0; seed < 2000; ++seed) {
            const Hierarchy h(seed, ill);
            Factor f(h.rows.rows(), h.rows.cols());
            f.prepare(h.ends.size());
            f.factor(h.view(), Eigen::Array<bool, Eigen::Dynamic, 1>::Ones(h.rows.rows()), h.ends, h.noise());
            const auto reference = hlq::test::reference_solve(h.level_rows, h.level_targets);
            bool same_ranks = true;
            for (Index k = 0; k < h.ends.size(); ++k) {
                same_ranks = same_ranks && f.rank(k) == reference.levels[static_cast<std::size_t>(k)].rank;
            }
            if (!same_ranks) {
                ++differ;  // a singular value at the threshold: the two methods may decide differently
                continue;
            }
            const Eigen::VectorXd x = solve(f, h);
            const double scale = 1 + reference.x.norm();
            EXPECT_LT((x - reference.x).norm(), (ill ? 1e-6 : 1e-10) * scale) << "seed " << seed;
        }
        EXPECT_LE(differ, ill ? 10 : 0);
    }
}

TEST(PriorityLQ, MultipliersMatchTheReference) {
    for (unsigned seed = 0; seed < 2000; ++seed) {
        const Hierarchy h(seed);
        Factor f(h.rows.rows(), h.rows.cols());
        f.prepare(h.ends.size());
        f.factor(h.view(), Eigen::Array<bool, Eigen::Dynamic, 1>::Ones(h.rows.rows()), h.ends, h.noise());
        Factor::RowsVector goal(f.size()), residual(f.size()), dual(f.size());
        Factor::ColsVector step(h.rows.cols());
        for (Index p = 0; p < f.size(); ++p) {
            goal(p) = h.targets(f.row(p));
        }
        f.solve(goal, residual, step);
        const auto reference = hlq::test::reference_solve(h.level_rows, h.level_targets);
        for (Index level = 1; level < h.ends.size(); ++level) {
            const auto lambda = hlq::test::reference_multipliers(reference, h.level_rows, h.level_targets,
                                                                 static_cast<std::size_t>(level));
            f.multipliers(level, residual, dual);
            for (Index k = 0; k <= level; ++k) {
                const Index begin = k == 0 ? 0 : h.ends(k - 1);
                for (Index i = begin; i < h.ends(k); ++i) {
                    const double expected = lambda[static_cast<std::size_t>(k)](i - begin);
                    EXPECT_NEAR(dual(f.position(i)), expected,
                                1e-8 * (1 + lambda[static_cast<std::size_t>(k)].norm()))
                        << "seed " << seed << " level " << level << " row " << i;
                }
            }
        }
    }
}

TEST(PriorityLQ, UpdatesMatchAFreshFactorization) {
    for (unsigned seed = 0; seed < 500; ++seed) {
        const Hierarchy h(seed);
        std::mt19937 gen(seed + 12345u);
        const Index m = h.rows.rows();
        Eigen::Array<bool, Eigen::Dynamic, 1> active = Eigen::Array<bool, Eigen::Dynamic, 1>::Zero(m);
        for (Index i = 0; i < m; ++i) {
            active(i) = std::uniform_int_distribution<int>(0, 1)(gen) == 1;
        }
        const auto noise = h.noise();
        Factor f(m, h.rows.cols());
        f.prepare(h.ends.size());
        f.factor(h.view(), active, h.ends, noise);
        for (int change = 0; change < 12; ++change) {
            const auto i = std::uniform_int_distribution<Index>(0, m - 1)(gen);
            Index level = 0;
            while (h.ends(level) <= i) {
                ++level;
            }
            if (active(i)) {
                f.remove(h.view(), i, noise);
            } else {
                f.insert(h.view(), i, level, noise);
            }
            active(i) = !active(i);
            expect_consistent(f, h.rows);

            Factor fresh(m, h.rows.cols());
            fresh.prepare(h.ends.size());
            fresh.factor(h.view(), active, h.ends, noise);
            for (Index k = 0; k < h.ends.size(); ++k) {
                EXPECT_EQ(f.rank(k), fresh.rank(k)) << "seed " << seed << " change " << change;
            }
            const Eigen::VectorXd x = solve(f, h);
            const Eigen::VectorXd y = solve(fresh, h);
            EXPECT_LT((x - y).norm(), 1e-9 * (1 + y.norm())) << "seed " << seed << " change " << change;
        }
    }
}

TEST(PriorityLQ, RowsInLaterBlocksWithoutDependentRowsGiveTheSameSolutionAndMultipliers) {
    // Rows inserted with a limit may sit in a later block than their level's; the solution, the ranks of the
    // levels and the multipliers of the final choice must still be those of a fresh factorization, which
    // keeps every row in its level's block.
    int displaced = 0;
    for (unsigned seed = 0; seed < 500; ++seed) {
        const Hierarchy h(seed);
        std::mt19937 gen(seed + 54321u);
        const Index m = h.rows.rows();
        const Index levels = h.ends.size();
        Eigen::Array<bool, Eigen::Dynamic, 1> active = Eigen::Array<bool, Eigen::Dynamic, 1>::Zero(m);
        for (Index i = 0; i < m; ++i) {
            active(i) = std::uniform_int_distribution<int>(0, 2)(gen) == 0;
        }
        const auto noise = h.noise();
        Factor f(m, h.rows.cols());
        f.prepare(levels);
        f.factor(h.view(), active, h.ends, noise);
        for (int change = 0; change < 12; ++change) {
            const auto i = std::uniform_int_distribution<Index>(0, m - 1)(gen);
            Index level = 0;
            while (h.ends(level) <= i) {
                ++level;
            }
            if (active(i)) {
                f.remove(h.view(), i, noise);
            } else {
                f.insert(h.view(), i, level, noise, levels);
            }
            active(i) = !active(i);
            displaced += f.displaced() ? 1 : 0;
            expect_consistent(f, h.rows);

            Factor fresh(m, h.rows.cols());
            fresh.prepare(levels);
            fresh.factor(h.view(), active, h.ends, noise);
            for (Index k = 0; k < levels; ++k) {
                EXPECT_EQ(f.rank(k), fresh.rank(k)) << "seed " << seed << " change " << change;
            }
            const Eigen::VectorXd x = solve(f, h);
            const Eigen::VectorXd y = solve(fresh, h);
            ASSERT_LT((x - y).norm(), 1e-9 * (1 + y.norm())) << "seed " << seed << " change " << change;

            Factor::RowsVector residual(m), dual(m), fresh_residual(m), fresh_dual(m);
            Factor::RowsVector goal(m);
            Factor::ColsVector step(h.rows.cols());
            for (Index p = 0; p < f.size(); ++p) {
                goal(p) = h.targets(f.row(p));
            }
            f.solve(goal, residual, step);
            f.multipliers(f.factored(), residual, dual);
            for (Index p = 0; p < fresh.size(); ++p) {
                goal(p) = h.targets(fresh.row(p));
            }
            fresh.solve(goal, fresh_residual, step);
            fresh.multipliers(fresh.factored(), fresh_residual, fresh_dual);
            double size = 0;
            for (Index p = 0; p < fresh.size(); ++p) {
                size = std::max(size, std::abs(fresh_dual(p)));
            }
            for (Index p = 0; p < fresh.size(); ++p) {
                EXPECT_NEAR(dual(f.position(fresh.row(p))), fresh_dual(p), 1e-8 * (1 + size))
                    << "seed " << seed << " change " << change << " row " << fresh.row(p);
            }
        }
    }
    EXPECT_GT(displaced, 500);
}
