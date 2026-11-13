// Compile-time capacities and single precision.

#include <gtest/gtest.h>

#include <hlq/eigen.hpp>
#include <hlq/hlq.hpp>

#include "support/certificate.hpp"

namespace {

using hlq::test::Family;

}  // namespace

TEST(Capacity, FixedSizeSolverMatchesTheDynamicOne) {
    // Random problems have at most 6 * 5 + 8 rows, 8 variables and 7 levels.
    using Fixed = hlq::Solver<double, 38, 8, 7>;
    static_assert(std::is_default_constructible_v<Fixed>);
    for (unsigned seed = 0; seed < 1000; ++seed) {
        const auto p = hlq::test::make_problem(Family::Random, seed);
        Fixed fixed(p.rows(), p.cols());
        hlq::Solver<> dynamic(p.rows(), p.cols());
        ASSERT_EQ(fixed.set_problem(p.matrix, p.lower, p.upper, p.level_ends), hlq::Status::Ready);
        ASSERT_EQ(dynamic.set_problem(p.matrix, p.lower, p.upper, p.level_ends), hlq::Status::Ready);
        ASSERT_EQ(fixed.solve(), hlq::Status::Solved);
        ASSERT_EQ(dynamic.solve(), hlq::Status::Solved);
        EXPECT_EQ(fixed.changes(), dynamic.changes());
        EXPECT_LT(
            (Eigen::VectorXd(hlq::as_eigen(fixed.solution())) - hlq::as_eigen(dynamic.solution())).norm(),
            1e-14 * (1 + hlq::as_eigen(dynamic.solution()).norm()));
    }
}

TEST(Capacity, FixedSizeSolverRejectsTooManyLevels) {
    hlq::Solver<double, 4, 2, 2> solver;
    const Eigen::MatrixXd a = Eigen::MatrixXd::Identity(3, 2);
    const Eigen::VectorXd b = Eigen::VectorXd::Zero(3);
    EXPECT_EQ(solver.set_problem(a, b, b, Eigen::Vector3i(1, 2, 3)), hlq::Status::InvalidLevels);
    EXPECT_EQ(solver.set_problem(a, b, b, Eigen::Vector2i(1, 3)), hlq::Status::Ready);
}

TEST(Capacity, SinglePrecision) {
    for (unsigned seed = 0; seed < 1000; ++seed) {
        const auto p = hlq::test::make_problem(Family::Integer, seed);
        hlq::Solver<float> single(p.rows(), p.cols());
        hlq::Solver<double> full(p.rows(), p.cols());
        ASSERT_EQ(single.set_problem(p.matrix, p.lower, p.upper, p.level_ends), hlq::Status::Ready);
        ASSERT_EQ(full.set_problem(p.matrix, p.lower, p.upper, p.level_ends), hlq::Status::Ready);
        ASSERT_EQ(single.solve(), hlq::Status::Solved) << "seed " << seed;
        ASSERT_EQ(full.solve(), hlq::Status::Solved);
        const Eigen::VectorXd x = hlq::as_eigen(single.solution()).cast<double>();
        EXPECT_LT((x - hlq::as_eigen(full.solution())).norm(),
                  1e-4 * (1 + hlq::as_eigen(full.solution()).norm()))
            << "seed " << seed;
        const auto cert = hlq::test::certify(p, x, {}, {}, 1e-4, 1e-3);
        EXPECT_TRUE(cert.optimal) << "seed " << seed << ": level " << cert.level << " unbalanced by "
                                  << cert.residual;
    }
}
