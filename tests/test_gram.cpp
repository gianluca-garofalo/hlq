// Gram mode (Settings::gram_mode) keeps only K, not Q. On well-conditioned problems it must give what the
// default gives: the same ranks, the solution to well within the tolerances, and optimality.

#include <random>
#include <string>

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

struct Outcome {
    Status status;
    Eigen::VectorXd x;
    std::vector<Index> ranks;
    std::vector<hlq::Activity> activity;
};

Outcome solve(const Problem& p, bool gram, const Eigen::MatrixXd& metric = {},
              const Eigen::VectorXd& reference = {}) {
    hlq::Solver<> solver(p.rows(), p.cols());
    solver.settings().gram_mode = gram;
    EXPECT_EQ(solver.set_problem(p.matrix, p.lower, p.upper, p.level_ends), Status::Ready);
    if (metric.size() > 0) {
        EXPECT_EQ(solver.set_metric(metric), Status::Ready);
        EXPECT_EQ(solver.set_reference(reference), Status::Ready);
    }
    Outcome o{solver.solve(), hlq::as_eigen(solver.solution()), {}, {}};
    for (Index k = 0; k < p.levels(); ++k) {
        o.ranks.push_back(solver.rank(k));
    }
    for (Index i = 0; i < p.rows(); ++i) {
        o.activity.push_back(solver.activity(i));
    }
    return o;
}

void expect_default(const Problem& p, const std::string& label, const Eigen::MatrixXd& metric = {},
                    const Eigen::VectorXd& reference = {}) {
    const Outcome gram = solve(p, true, metric, reference);
    const Outcome lq = solve(p, false, metric, reference);
    ASSERT_EQ(gram.status, Status::Solved) << label;
    ASSERT_EQ(lq.status, Status::Solved) << label;
    if (gram.activity == lq.activity) {  // a degenerate problem can have more than one optimal active set
        EXPECT_EQ(gram.ranks, lq.ranks) << label;
    }
    EXPECT_LT((gram.x - lq.x).norm(), 1e-6 * (1 + lq.x.norm())) << label;
    const auto cert = hlq::test::certify(p, gram.x, metric, reference);
    EXPECT_TRUE(cert.optimal) << label << ": level " << cert.level << " unbalanced by " << cert.residual;
}

}  // namespace

TEST(Gram, MatchesTheDefaultOnEveryFamily) {
    for (const Family family : {Family::Random, Family::Integer, Family::Degenerate, Family::Large}) {
        for (unsigned seed = 0; seed < (family == Family::Large ? 100u : 500u); ++seed) {
            expect_default(hlq::test::make_problem(family, seed),
                           std::string(hlq::test::to_string(family)) + " seed " + std::to_string(seed));
        }
    }
}

TEST(Gram, MatchesTheDefaultInAMetric) {
    for (unsigned seed = 0; seed < 200; ++seed) {
        const Problem p = hlq::test::make_problem(Family::Random, seed);
        const Eigen::MatrixXd w = hlq::test::make_metric(p.cols(), seed);
        const Eigen::VectorXd r = Eigen::VectorXd::LinSpaced(p.cols(), -1, 1);
        expect_default(p, "seed " + std::to_string(seed), w, r);
    }
}

TEST(Gram, RanksStopAtTheColumns) {
    // Level 0 has rank 4, every column: rows (1, t, t^2, t^3) for t = 1/8 to 4/8, conditioned 1e3. What a
    // row of ones at level 1 has outside them, computed from inner products, is rounding noise above the rank
    // threshold: it must not count.
    Eigen::Matrix<double, 5, 4> a;
    Eigen::Matrix<double, 5, 1> b;
    for (int i = 0; i < 4; ++i) {
        const double t = (i + 1) / 8.0;
        a.row(i) << 1, t, t * t, t * t * t;
        b(i) = i + 1;
    }
    a.row(4).setOnes();
    b(4) = 0;
    hlq::Solver<> solver(5, 4);
    solver.settings().gram_mode = true;
    solver.settings().fold_final_level = false;
    ASSERT_EQ(solver.set_problem(a, b, b, {4, 5}), Status::Ready);
    ASSERT_EQ(solver.solve(), Status::Solved);
    EXPECT_EQ(solver.rank(0), 4);
    EXPECT_EQ(solver.rank(1), 0);
}

TEST(Gram, WarmSolvesAfterTheBoundsMove) {
    for (unsigned seed = 0; seed < 100; ++seed) {
        Problem p = hlq::test::make_problem(Family::Random, seed);
        std::mt19937 gen(seed + 4242u);
        std::uniform_real_distribution<double> shift(-0.2, 0.2);
        hlq::Solver<> solver(p.rows(), p.cols());
        solver.settings().gram_mode = true;
        for (int step = 0; step < 6; ++step) {
            const std::string label = "seed " + std::to_string(seed) + " step " + std::to_string(step);
            ASSERT_EQ(solver.set_problem(p.matrix, p.lower, p.upper, p.level_ends), Status::Ready);
            ASSERT_EQ(solver.solve(), Status::Solved) << label;
            const Eigen::VectorXd x = hlq::as_eigen(solver.solution());
            const Outcome cold = solve(p, true);
            EXPECT_LT((x - cold.x).norm(), 1e-6 * (1 + cold.x.norm())) << label;
            const auto cert = hlq::test::certify(p, x);
            EXPECT_TRUE(cert.optimal)
                << label << ": level " << cert.level << " unbalanced by " << cert.residual;
            for (Index i = 0; i < p.rows() - p.cols(); ++i) {
                const double d = shift(gen);
                p.lower(i) += d;
                p.upper(i) += d;
            }
        }
    }
}

TEST(Gram, SwitchingTheModeFactorsAgain) {
    for (unsigned seed = 0; seed < 50; ++seed) {
        const Problem p = hlq::test::make_problem(Family::Large, seed);
        hlq::Solver<> solver(p.rows(), p.cols());
        ASSERT_EQ(solver.set_problem(p.matrix, p.lower, p.upper, p.level_ends), Status::Ready);
        for (const bool gram : {false, true, false}) {
            solver.settings().gram_mode = gram;
            ASSERT_EQ(solver.solve(), Status::Solved);
            const Outcome fresh = solve(p, gram);
            EXPECT_LT((hlq::as_eigen(solver.solution()) - fresh.x).norm(), 1e-9 * (1 + fresh.x.norm()))
                << "seed " << seed << (gram ? " gram" : " default");
        }
    }
}
