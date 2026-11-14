// A last level of equalities with full column rank, taken as the choice among the optima (Settings::
// fold_final_level), gives what searching it gives: the solution, every level's cost and rank, and the
// multipliers of every subproblem, the choice among the optima included.

#include <random>

#include <gtest/gtest.h>

#include <hlq/eigen.hpp>
#include <hlq/hlq.hpp>

#include "support/certificate.hpp"
#include "support/problems.hpp"

namespace {

using hlq::Status;
using hlq::test::Family;
using hlq::test::Problem;
using Index = Eigen::Index;

/// A problem of the family whose last level (x = 0) is replaced by `extra` more random equalities than
/// variables, which then have full column rank.
Problem with_random_last_level(Problem p, unsigned seed, Index extra) {
    std::mt19937 gen(seed + 31337u);
    std::normal_distribution<double> normal;
    const Index n = p.cols();
    const Index above = p.rows() - n;
    Problem q{Eigen::MatrixXd(above + n + extra, n), Eigen::VectorXd(above + n + extra),
              Eigen::VectorXd(above + n + extra), p.level_ends};
    q.matrix.topRows(above) = p.matrix.topRows(above);
    q.lower.head(above) = p.lower.head(above);
    q.upper.head(above) = p.upper.head(above);
    for (Index i = above; i < above + n + extra; ++i) {
        for (Index j = 0; j < n; ++j) {
            q.matrix(i, j) = normal(gen);
        }
        q.lower(i) = q.upper(i) = normal(gen);
    }
    q.level_ends(q.levels() - 1) = static_cast<int>(above + n + extra);
    return q;
}

struct Outcome {
    Status status;
    Eigen::VectorXd x;
    std::vector<double> costs;
    std::vector<Index> ranks;
    std::vector<Eigen::VectorXd> multipliers;
    std::vector<hlq::Activity> activity;
};

Outcome solve(const Problem& p, bool fold, const Eigen::MatrixXd& metric, const Eigen::VectorXd& reference) {
    hlq::Solver<> solver(p.rows(), p.cols());
    solver.settings().fold_final_level = fold;
    EXPECT_EQ(solver.set_problem(p.matrix, p.lower, p.upper, p.level_ends), Status::Ready);
    if (metric.size() > 0) {
        EXPECT_EQ(solver.set_metric(metric), Status::Ready);
        EXPECT_EQ(solver.set_reference(reference), Status::Ready);
    }
    Outcome o{solver.solve(), hlq::as_eigen(solver.solution()), {}, {}, {}, {}};
    for (Index k = 0; k < p.levels(); ++k) {
        o.costs.push_back(solver.level_cost(k));
        o.ranks.push_back(solver.rank(k));
    }
    // Every subproblem, the last one twice: multipliers(levels()) brings the folded level into the
    // factorization, and the others must not see it.
    for (Index k = 0; k <= p.levels(); ++k) {
        o.multipliers.emplace_back(hlq::as_eigen(solver.multipliers(k)));
    }
    o.multipliers.emplace_back(hlq::as_eigen(solver.multipliers(p.levels() - 1)));
    for (Index i = 0; i < p.rows(); ++i) {
        o.activity.push_back(solver.activity(i));
    }
    return o;
}

void expect_same(const Problem& p, const Eigen::MatrixXd& metric, const Eigen::VectorXd& reference,
                 const std::string& label) {
    const Outcome folded = solve(p, true, metric, reference);
    const Outcome searched = solve(p, false, metric, reference);
    ASSERT_EQ(folded.status, Status::Solved) << label;
    ASSERT_EQ(searched.status, Status::Solved) << label;
    const double scale = 1 + searched.x.norm();
    EXPECT_LT((folded.x - searched.x).norm(), 1e-10 * scale) << label;
    for (std::size_t k = 0; k < folded.costs.size(); ++k) {
        EXPECT_NEAR(folded.costs[k], searched.costs[k], 1e-9 * (1 + searched.costs[k]))
            << label << " level " << k;
        EXPECT_EQ(folded.ranks[k], searched.ranks[k]) << label << " level " << k;
    }
    const auto cert = hlq::test::certify(p, folded.x, metric, reference);
    EXPECT_TRUE(cert.optimal) << label << ": level " << cert.level << " unbalanced by " << cert.residual;
    // Multipliers are those of the active set, which degenerate problems may choose differently.
    if (folded.activity != searched.activity) {
        return;
    }
    for (std::size_t k = 0; k < folded.multipliers.size(); ++k) {
        const double size = 1 + searched.multipliers[k].norm();
        EXPECT_LT((folded.multipliers[k] - searched.multipliers[k]).norm(), 1e-8 * size)
            << label << " subproblem " << k;
    }
}

}  // namespace

TEST(Fold, MatchesTheSearchOnEveryFamily) {
    for (const Family family : {Family::Random, Family::Integer, Family::Degenerate, Family::Large}) {
        for (unsigned seed = 0; seed < (family == Family::Large ? 100u : 500u); ++seed) {
            expect_same(hlq::test::make_problem(family, seed), {}, {},
                        std::string(hlq::test::to_string(family)) + " seed " + std::to_string(seed));
        }
    }
}

TEST(Fold, MatchesTheSearchWithARandomLastLevel) {
    for (unsigned seed = 0; seed < 500; ++seed) {
        const Problem p =
            with_random_last_level(hlq::test::make_problem(Family::Random, seed), seed, seed % 3);
        expect_same(p, {}, {}, "seed " + std::to_string(seed));
    }
}

TEST(Fold, MatchesTheSearchWithAMetricAndAReference) {
    for (unsigned seed = 0; seed < 500; ++seed) {
        const Problem p =
            seed % 2 == 0 ? hlq::test::make_problem(Family::Random, seed)
                          : with_random_last_level(hlq::test::make_problem(Family::Random, seed), seed, 1);
        const Eigen::MatrixXd w = hlq::test::make_metric(p.cols(), seed);
        const Eigen::VectorXd r = Eigen::VectorXd::LinSpaced(p.cols(), -1, 1);
        expect_same(p, w, r, "seed " + std::to_string(seed));
    }
}

TEST(Fold, OnlyLevelOfEqualities) {
    // One level, x = (1, 2) and x1 + x2 = 0: its least squares, and every multiplier its residual.
    Eigen::Matrix<double, 3, 2> a;
    a << 1, 0, 0, 1, 1, 1;
    const Eigen::Vector3d b(1, 2, 0);
    hlq::Solver<> solver(3, 2);
    ASSERT_EQ(solver.set_problem(a, b, b, {3}), Status::Ready);
    ASSERT_EQ(solver.solve(), Status::Solved);
    const Eigen::Vector2d x = a.colPivHouseholderQr().solve(b);
    EXPECT_LT((hlq::as_eigen(solver.solution()) - x).norm(), 1e-14);
    EXPECT_EQ(solver.rank(0), 2);
    EXPECT_LT((hlq::as_eigen(solver.multipliers(0)) - (a * x - b)).norm(), 1e-14);
}

TEST(Fold, LeavesOtherLastLevelsToTheSearch) {
    // A last level with an inequality, and one without full column rank: both still solved exactly.
    for (unsigned seed = 0; seed < 300; ++seed) {
        Problem p = hlq::test::make_problem(Family::Random, seed);
        const Index last = p.rows() - 1;
        if (seed % 2 == 0) {
            p.upper(last) = 1;  // x_n in [0, 1]
        } else {
            p.matrix.row(last) = p.matrix.row(last - 1);  // two equal rows: rank n - 1
            p.lower(last) = p.upper(last) = 1;
        }
        hlq::Solver<> solver(p.rows(), p.cols());
        ASSERT_EQ(solver.set_problem(p.matrix, p.lower, p.upper, p.level_ends), Status::Ready);
        ASSERT_EQ(solver.solve(), Status::Solved) << "seed " << seed;
        const auto cert = hlq::test::certify(p, hlq::as_eigen(solver.solution()));
        EXPECT_TRUE(cert.optimal) << "seed " << seed << ": level " << cert.level << " unbalanced by "
                                  << cert.residual;
    }
}
