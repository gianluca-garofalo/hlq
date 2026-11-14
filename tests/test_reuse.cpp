// A solve keeps the factorization of its final active set for the next one, which reuses it when only the
// bounds or the reference changed. The results must be those of a solver that factors from scratch.

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

/// The bounds moved by up to `size` times their spread, keeping equalities equal.
void move_bounds(Problem& p, std::mt19937& gen, double size) {
    std::uniform_real_distribution<double> shift(-size, size);
    for (Index i = 0; i < p.rows(); ++i) {
        const double d = shift(gen) * (1 + std::abs(p.lower(i)));
        if (std::isfinite(p.lower(i))) {
            p.lower(i) += d;
        }
        if (std::isfinite(p.upper(i))) {
            p.upper(i) += d;
        }
    }
}

Eigen::VectorXd cold(const Problem& p) {
    hlq::Solver<> solver(p.rows(), p.cols());
    EXPECT_EQ(solver.set_problem(p.matrix, p.lower, p.upper, p.level_ends), Status::Ready);
    EXPECT_EQ(solver.solve(), Status::Solved);
    return hlq::as_eigen(solver.solution());
}

void expect_cold(const hlq::Solver<>& solver, const Problem& p, const std::string& label) {
    const Eigen::VectorXd x = hlq::as_eigen(solver.solution());
    const Eigen::VectorXd y = cold(p);
    EXPECT_LT((x - y).norm(), 1e-9 * (1 + y.norm())) << label;
    const auto cert = hlq::test::certify(p, x);
    EXPECT_TRUE(cert.optimal) << label << ": level " << cert.level << " unbalanced by " << cert.residual;
}

}  // namespace

TEST(Reuse, WarmSolvesAfterTheBoundsMoveMatchColdOnes) {
    for (const Family family : {Family::Random, Family::Degenerate, Family::Large}) {
        for (unsigned seed = 0; seed < (family == Family::Large ? 50u : 200u); ++seed) {
            Problem p = hlq::test::make_problem(family, seed);
            std::mt19937 gen(seed + 777u);
            hlq::Solver<> solver(p.rows(), p.cols());
            for (int step = 0; step < 8; ++step) {
                ASSERT_EQ(solver.set_problem(p.matrix, p.lower, p.upper, p.level_ends), Status::Ready);
                ASSERT_EQ(solver.solve(), Status::Solved);
                expect_cold(solver, p,
                            std::string(hlq::test::to_string(family)) + " seed " + std::to_string(seed)
                                + " step " + std::to_string(step));
                move_bounds(p, gen, step % 2 == 0 ? 0.01 : 0.2);
            }
        }
    }
}

TEST(Reuse, AWarmSolveOfTheSameProblemNeedsNoChange) {
    for (unsigned seed = 0; seed < 200; ++seed) {
        const Problem p = hlq::test::make_problem(Family::Random, seed);
        hlq::Solver<> solver(p.rows(), p.cols());
        ASSERT_EQ(solver.set_problem(p.matrix, p.lower, p.upper, p.level_ends), Status::Ready);
        ASSERT_EQ(solver.solve(), Status::Solved);
        const Eigen::VectorXd first = hlq::as_eigen(solver.solution());
        ASSERT_EQ(solver.set_problem(p.matrix, p.lower, p.upper, p.level_ends), Status::Ready);
        ASSERT_EQ(solver.solve(), Status::Solved);
        EXPECT_EQ(solver.changes(), 0) << "seed " << seed;
        EXPECT_LT((hlq::as_eigen(solver.solution()) - first).norm(), 1e-12 * (1 + first.norm()))
            << "seed " << seed;
    }
}

TEST(Reuse, EveryChangeOfTheRowsIsSeen) {
    // Each of these changes what the factorization depends on, between two solves of one solver.
    for (unsigned seed = 0; seed < 100; ++seed) {
        Problem p = hlq::test::make_problem(Family::Random, seed);
        hlq::Solver<> solver(p.rows(), p.cols());
        ASSERT_EQ(solver.set_problem(p.matrix, p.lower, p.upper, p.level_ends), Status::Ready);
        ASSERT_EQ(solver.solve(), Status::Solved);
        const std::string label = "seed " + std::to_string(seed);

        p.matrix(0, 0) += 0.5;  // one entry of the rows
        ASSERT_EQ(solver.set_problem(p.matrix, p.lower, p.upper, p.level_ends), Status::Ready);
        ASSERT_EQ(solver.solve(), Status::Solved);
        expect_cold(solver, p, label + " entry");

        const Eigen::MatrixXd w = hlq::test::make_metric(p.cols(), seed);  // the metric
        ASSERT_EQ(solver.set_metric(w), Status::Ready);
        ASSERT_EQ(solver.solve(), Status::Solved);
        const Eigen::VectorXd with_metric = hlq::as_eigen(solver.solution());
        hlq::Solver<> fresh(p.rows(), p.cols());
        ASSERT_EQ(fresh.set_problem(p.matrix, p.lower, p.upper, p.level_ends), Status::Ready);
        ASSERT_EQ(fresh.set_metric(w), Status::Ready);
        ASSERT_EQ(fresh.solve(), Status::Solved);
        EXPECT_LT((with_metric - hlq::as_eigen(fresh.solution())).norm(), 1e-9 * (1 + with_metric.norm()))
            << label;
        ASSERT_EQ(solver.set_metric(Eigen::MatrixXd::Identity(p.cols(), p.cols())), Status::Ready);

        solver.reset();  // the active set
        ASSERT_EQ(solver.solve(), Status::Solved);
        expect_cold(solver, p, label + " reset");

        solver.settings().rank_tolerance = 1e-6;  // the rank thresholds
        ASSERT_EQ(solver.solve(), Status::Solved);
        solver.settings().rank_tolerance = 1e-9;
        ASSERT_EQ(solver.solve(), Status::Solved);
        expect_cold(solver, p, label + " rank tolerance");

        solver.settings().fold_final_level = false;  // the levels searched
        ASSERT_EQ(solver.solve(), Status::Solved);
        expect_cold(solver, p, label + " no fold");
        solver.settings().fold_final_level = true;

        const Index i = p.level_ends(0) - 1;  // a bound becomes an equality, and back
        const double lower = p.lower(i), upper = p.upper(i);
        p.lower(i) = p.upper(i) = std::isfinite(upper) ? upper : lower;
        ASSERT_EQ(solver.set_problem(p.matrix, p.lower, p.upper, p.level_ends), Status::Ready);
        ASSERT_EQ(solver.solve(), Status::Solved);
        expect_cold(solver, p, label + " equality");
        p.lower(i) = lower;
        p.upper(i) = upper;
        ASSERT_EQ(solver.set_problem(p.matrix, p.lower, p.upper, p.level_ends), Status::Ready);
        ASSERT_EQ(solver.solve(), Status::Solved);
        expect_cold(solver, p, label + " inequality");
    }
}

TEST(Reuse, ManyWarmSolvesKeepTheirAccuracy) {
    // The factorization is updated, never refactored, across 2000 solves whose bounds move: no drift.
    for (const Family family : {Family::Random, Family::Large}) {
        for (unsigned seed = 0; seed < 5; ++seed) {
            const Problem base = hlq::test::make_problem(family, seed);
            Problem p = base;
            std::mt19937 gen(seed);
            std::uniform_real_distribution<double> shift(-0.05, 0.05);
            hlq::Solver<> solver(p.rows(), p.cols());
            for (int step = 0; step < 2000; ++step) {
                for (Index i = 0; i < p.rows(); ++i) {
                    const double d = shift(gen);
                    p.lower(i) = base.lower(i) + d;
                    p.upper(i) = base.upper(i) + d;
                }
                ASSERT_EQ(solver.set_problem(p.matrix, p.lower, p.upper, p.level_ends), Status::Ready);
                ASSERT_EQ(solver.solve(), Status::Solved);
            }
            const Eigen::VectorXd x = hlq::as_eigen(solver.solution());
            const Eigen::VectorXd y = cold(p);
            EXPECT_LT((x - y).norm(), 1e-12 * (1 + y.norm()))
                << hlq::test::to_string(family) << " seed " << seed;
        }
    }
}
