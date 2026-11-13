// The solver on small problems with known solutions, and its interface.

#include <limits>

#include <gtest/gtest.h>

#include <hlq/hlq.hpp>

namespace {

using hlq::Activity;
using hlq::Status;
using Index = Eigen::Index;
using Solver = hlq::Solver<>;

constexpr double inf = std::numeric_limits<double>::infinity();

struct Case {
    Eigen::MatrixXd matrix;
    Eigen::VectorXd lower;
    Eigen::VectorXd upper;
    Eigen::VectorXi ends;
};

/// Solves cold, then warm: the warm solve must give the same point with no changes.
Eigen::VectorXd solve_twice(const Case& c) {
    Solver solver(c.matrix.rows(), c.matrix.cols());
    EXPECT_EQ(solver.set_problem(c.matrix, c.lower, c.upper, c.ends), Status::Ready);
    EXPECT_EQ(solver.solve(), Status::Solved);
    const Eigen::VectorXd cold = solver.solution();
    EXPECT_EQ(solver.set_problem(c.matrix, c.lower, c.upper, c.ends), Status::Ready);
    EXPECT_EQ(solver.solve(), Status::Solved);
    EXPECT_EQ(solver.changes(), 0);
    EXPECT_LT((solver.solution() - cold).norm(), 1e-12 * (1 + cold.norm()));
    return cold;
}

}  // namespace

TEST(Solver, ConflictingLevelAndDependentRow) {
    // Level 1: x1 + x2 = 2. Level 2: x1 = 2, x2 = 1 (conflicting under level 1). Level 3: x1 = 0 (no freedom
    // left for it), x3 = 3.
    Case c{Eigen::MatrixXd(5, 3), Eigen::VectorXd(5), Eigen::VectorXd(5), Eigen::VectorXi(3)};
    c.matrix << 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1;
    c.lower << 2, 2, 1, 0, 3;
    c.upper = c.lower;
    c.ends << 1, 3, 5;
    EXPECT_TRUE(solve_twice(c).isApprox(Eigen::Vector3d(1.5, 0.5, 3)));
}

TEST(Solver, ActiveBounds) {
    // Level 1: -1 <= x <= 1. Level 2: x1 + x2 + x3 <= 1. Level 3: x1 - x2 = 0.5. Level 4: 10 <= 3 x1 + x2 -
    // x3.
    Case c{Eigen::MatrixXd(6, 3), Eigen::VectorXd(6), Eigen::VectorXd(6), Eigen::VectorXi(4)};
    c.matrix << Eigen::Matrix3d::Identity(), Eigen::RowVector3d(1, 1, 1), Eigen::RowVector3d(1, -1, 0),
        Eigen::RowVector3d(3, 1, -1);
    c.lower << -1, -1, -1, -inf, 0.5, 10;
    c.upper << 1, 1, 1, 1, 0.5, inf;
    c.ends << 3, 4, 5, 6;
    EXPECT_TRUE(solve_twice(c).isApprox(Eigen::Vector3d(1, 0.5, -1)));
}

TEST(Solver, EqualityAfterAnInactiveRow) {
    // An equality after an inactive row of its level stays active against the level below.
    Case c{Eigen::MatrixXd(4, 2), Eigen::VectorXd(4), Eigen::VectorXd(4), Eigen::VectorXi(2)};
    c.matrix << 1, -1, 1, 1, 1, 0, 0, 1;
    c.lower << -10, 1, -3, -3;
    c.upper << 10, 1, -3, -3;
    c.ends << 2, 4;
    EXPECT_TRUE(solve_twice(c).isApprox(Eigen::Vector2d(0.5, 0.5)));
}

TEST(Solver, StopsAtTheFirstBlockingConstraint) {
    // Seed 1633 of the 04_random_problem generator, reduced: stepping past the inactive rows of level 1 would
    // end with level 2 cost 6.42 instead of 5.806. Expected solution from LexLS.
    Case c{Eigen::MatrixXd(6, 2), Eigen::VectorXd(6), Eigen::VectorXd(6), Eigen::VectorXi(3)};
    c.matrix << 0.94938431195970785, 0.31411690213166171, -0.34438211051399187, -0.9388295702404823,
        0.81036629268301097, -0.58592360567158641, 0.69931374302526361, 0.71481486331496746, 1, 0, 0, 1;
    c.lower << -2.8907654755895202, -1.1453187756329379, -0.80526566693063739, 3.293845404453144, 0, 0;
    c.upper << 0.40657261602089423, 0.035593689003228367, -0.46677367697495692, 4.2613014038087895, 0, 0;
    c.ends << 3, 4, 6;
    EXPECT_TRUE(solve_twice(c).isApprox(Eigen::Vector2d(0.0280136139523878, 1.20966725392553)));
}

TEST(Solver, LowerLevelReleasesABound) {
    // Level 1: 0.5 <= x2 <= 1.5. Level 2: 3 <= x2 <= 4 (x2 = 1.5 is the best it can do). Level 3: x = 0.
    Case c{Eigen::MatrixXd(6, 2), Eigen::VectorXd(6), Eigen::VectorXd(6), Eigen::VectorXi(3)};
    c.matrix << 0, -2, 0, 2, 0, 1, 0, 1, 1, 0, 0, 1;
    c.lower << -4, -2, -3, 3, 0, 0;
    c.upper << -1, 3, 2, 4, 0, 0;
    c.ends << 3, 4, 6;
    EXPECT_TRUE(solve_twice(c).isApprox(Eigen::Vector2d(0, 1.5)));
}

TEST(Solver, ParallelRowsGetNoRankFromRoundingNoise) {
    Case c{Eigen::MatrixXd(6, 2), Eigen::VectorXd(6), Eigen::VectorXd(6), Eigen::VectorXi(3)};
    c.matrix << -1, -1, 2, 2, -2, -2, 2, 2, 1, 0, 0, 1;
    c.lower << -3, -4, -3, -4, 0, 0;
    c.upper << 4, -1, -1, -2, 0, 0;
    c.ends << 3, 4, 6;
    EXPECT_LT(solve_twice(c).norm(), 1e-12);
}

TEST(Solver, MultiplierTwoLevelsBelowReleasesABound) {
    // Level 1: x1 <= 1. Level 2: x1 + x2 = 4. Level 3: x2 = 2, then x2 = 4. The bound active from the first
    // solve is released by its multiplier in the subproblem of level 3.
    Eigen::Matrix<double, 3, 2> a;
    a << 1, 0, 1, 1, 0, 1;
    Eigen::Vector3d lower(-inf, 4, 2), upper(1, 4, 2);
    const Eigen::Vector3i ends(1, 2, 3);
    Solver solver(3, 2);
    ASSERT_EQ(solver.set_problem(a, lower, upper, ends), Status::Ready);
    ASSERT_EQ(solver.solve(), Status::Solved);
    EXPECT_TRUE(solver.solution().isApprox(Eigen::Vector2d(1, 3)));
    EXPECT_EQ(solver.activity(0), Activity::Upper);
    // The level-3 subproblem: its residual is x2 - 2 = 1, and through the equality of level 2 the bound of
    // level 1 holds it back with a nonnegative multiplier, as an upper bound should.
    const Eigen::VectorXd lambda = solver.multipliers(2);
    EXPECT_NEAR(lambda(2), 1, 1e-12);
    EXPECT_NEAR(lambda(1), -1, 1e-12);
    EXPECT_NEAR(lambda(0), 1, 1e-12);

    lower(2) = upper(2) = 4;
    ASSERT_EQ(solver.set_problem(a, lower, upper, ends), Status::Ready);
    ASSERT_EQ(solver.solve(), Status::Solved);
    EXPECT_LT((solver.solution() - Eigen::Vector2d(0, 4)).norm(), 1e-12);
    EXPECT_EQ(solver.changes(), 1);
    EXPECT_EQ(solver.activity(0), Activity::Inactive);
}

TEST(Solver, ReportsLevelsAtTheSolution) {
    // Level 1: x1 <= 1. Level 2: x1 = 2 and x2 = 1.
    Eigen::Matrix<double, 3, 2> a;
    a << 1, 0, 1, 0, 0, 1;
    const Eigen::Vector3d lower(-inf, 2, 1), upper(1, 2, 1);
    const Eigen::Vector2i ends(1, 3);
    Solver solver(3, 2);
    ASSERT_EQ(solver.set_problem(a, lower, upper, ends), Status::Ready);
    ASSERT_EQ(solver.solve(), Status::Solved);
    EXPECT_TRUE(solver.solution().isApprox(Eigen::Vector2d(1, 1)));
    EXPECT_EQ(solver.activity(0), Activity::Upper);
    EXPECT_EQ(solver.activity(1), Activity::Equality);
    EXPECT_NEAR(solver.level_cost(0), 0, 1e-24);
    EXPECT_NEAR(solver.level_cost(1), 1, 1e-12);
    EXPECT_NEAR(solver.violation()(1), -1, 1e-12);
    EXPECT_EQ(solver.rank(0), 1);
    EXPECT_EQ(solver.rank(1), 1);
    // An upper bound holding a level back has a nonnegative multiplier.
    EXPECT_NEAR(solver.multipliers(1)(0), 1, 1e-12);
}

TEST(Solver, ReferenceAndMetricChooseAmongTheOptima) {
    // One level, x1 + x2 = 2: a line of optima.
    const Eigen::RowVector2d a(1, 1);
    const Eigen::Matrix<double, 1, 1> b(2);
    const Eigen::Matrix<int, 1, 1> ends(1);
    Solver solver(1, 2);
    ASSERT_EQ(solver.set_problem(a, b, b, ends), Status::Ready);
    ASSERT_EQ(solver.solve(), Status::Solved);
    EXPECT_TRUE(solver.solution().isApprox(Eigen::Vector2d(1, 1)));  // minimum norm

    ASSERT_EQ(solver.set_reference(Eigen::Vector2d(3, 0)), Status::Ready);
    ASSERT_EQ(solver.solve(), Status::Solved);
    EXPECT_TRUE(solver.solution().isApprox(Eigen::Vector2d(2.5, -0.5)));  // closest to the reference

    ASSERT_EQ(solver.set_reference(Eigen::Vector2d::Zero()), Status::Ready);
    ASSERT_EQ(solver.set_metric(Eigen::Vector2d(1, 4).asDiagonal().toDenseMatrix()), Status::Ready);
    ASSERT_EQ(solver.solve(), Status::Solved);
    EXPECT_TRUE(solver.solution().isApprox(Eigen::Vector2d(1.6, 0.4)));  // minimum of x1^2 + 4 x2^2
}

TEST(Solver, InfiniteBoundsAreNeverActive) {
    Eigen::Matrix<double, 3, 2> a;
    a << 1, 1, 1, 0, 0, 1;
    const Eigen::Vector3d lower(-inf, -inf, 3), upper(inf, 2, inf);
    const Eigen::Vector2i ends(2, 3);
    Solver solver(3, 2);
    ASSERT_EQ(solver.set_problem(a, lower, upper, ends), Status::Ready);
    ASSERT_EQ(solver.solve(), Status::Solved);
    EXPECT_EQ(solver.activity(0), Activity::Inactive);
    EXPECT_TRUE(solver.solution().isApprox(Eigen::Vector2d(0, 3)));
}

TEST(Solver, RejectsInvalidProblems) {
    Solver solver(4, 2);
    EXPECT_EQ(solver.solve(), Status::NoProblem);
    const Eigen::MatrixXd a = Eigen::MatrixXd::Ones(3, 2);
    const Eigen::VectorXd l = Eigen::VectorXd::Zero(3), u = Eigen::VectorXd::Ones(3);
    EXPECT_EQ(solver.set_problem(Eigen::MatrixXd::Ones(3, 3), l, u, Eigen::Vector2i(1, 3)),
              Status::InvalidDimensions);
    EXPECT_EQ(solver.set_problem(Eigen::MatrixXd::Ones(5, 2), Eigen::VectorXd::Zero(5),
                                 Eigen::VectorXd::Ones(5), Eigen::Vector2i(1, 5)),
              Status::InvalidDimensions);
    EXPECT_EQ(solver.set_problem(a, l, u, Eigen::Vector2i(2, 1)), Status::InvalidLevels);
    EXPECT_EQ(solver.set_problem(a, l, u, Eigen::Vector2i(1, 2)), Status::InvalidLevels);
    EXPECT_EQ(solver.set_problem(a, l, u, Eigen::VectorXi()), Status::InvalidLevels);
    EXPECT_EQ(solver.set_problem(a, u, l, Eigen::Vector2i(1, 3)), Status::InvalidBounds);
    EXPECT_EQ(solver.set_problem(a, Eigen::Vector3d(0, std::nan(""), 0), u, Eigen::Vector2i(1, 3)),
              Status::InvalidBounds);
    EXPECT_EQ(
        solver.set_problem(a, Eigen::Vector3d(0, inf, 0), Eigen::Vector3d(1, inf, 1), Eigen::Vector2i(1, 3)),
        Status::InvalidBounds);
    Eigen::MatrixXd bad = a;
    bad(1, 1) = inf;
    EXPECT_EQ(solver.set_problem(bad, l, u, Eigen::Vector2i(1, 3)), Status::InvalidData);
    EXPECT_EQ(solver.solve(), Status::InvalidData);

    ASSERT_EQ(solver.set_problem(a, l, u, Eigen::Vector2i(1, 3)), Status::Ready);
    EXPECT_EQ(solver.set_metric(Eigen::Matrix2d::Identity() * -1), Status::InvalidMetric);
    EXPECT_EQ(solver.set_metric(Eigen::Matrix2d(Eigen::Matrix2d::Ones() + Eigen::Matrix2d::Identity())
                                    .triangularView<Eigen::Upper>()
                                    .toDenseMatrix()),
              Status::InvalidMetric);
    EXPECT_EQ(solver.set_metric(Eigen::Matrix3d::Identity()), Status::InvalidDimensions);
    EXPECT_EQ(solver.set_reference(Eigen::Vector3d::Zero()), Status::InvalidDimensions);
    EXPECT_EQ(solver.set_reference(Eigen::Vector2d(0, inf)), Status::InvalidData);
    EXPECT_EQ(solver.solve(), Status::Solved);
}

TEST(Solver, StopsAtMaxChanges) {
    Eigen::Matrix<double, 3, 2> a;
    a << 1, 0, 0, 1, 1, 1;
    const Eigen::Vector3d lower(-inf, -inf, 4), upper(1, 1, 4);
    const Eigen::Vector2i ends(2, 3);
    Solver solver(3, 2);
    solver.settings().max_changes = 1;
    ASSERT_EQ(solver.set_problem(a, lower, upper, ends), Status::Ready);
    EXPECT_EQ(solver.solve(), Status::MaxChanges);
    EXPECT_TRUE(hlq::has_solution(solver.status()));
    solver.settings().max_changes = 500;
    solver.reset();
    EXPECT_EQ(solver.solve(), Status::Solved);
    EXPECT_TRUE(solver.solution().isApprox(Eigen::Vector2d(1, 1)));
}

TEST(Solver, KeepsTheWarmStartOnlyForTheSameStructure) {
    Eigen::Matrix<double, 2, 1> a(1, 1);
    const Eigen::Vector2d lower(-inf, 3), upper(1, 3);
    Solver solver(2, 1);
    ASSERT_EQ(solver.set_problem(a, lower, upper, Eigen::Vector2i(1, 2)), Status::Ready);
    ASSERT_EQ(solver.solve(), Status::Solved);
    EXPECT_EQ(solver.activity(0), Activity::Upper);
    ASSERT_EQ(solver.set_problem(a, lower, upper, Eigen::Vector2i(1, 2)), Status::Ready);
    EXPECT_EQ(solver.activity(0), Activity::Upper);
    ASSERT_EQ(solver.set_problem(a.topRows(1), lower.head(1), upper.head(1), Eigen::Matrix<int, 1, 1>(1)),
              Status::Ready);
    EXPECT_EQ(solver.activity(0), Activity::Inactive);
}

TEST(Solver, StatusNames) {
    EXPECT_STREQ(hlq::to_string(Status::Solved), "solved");
    EXPECT_STREQ(hlq::to_string(Status::InvalidMetric), "metric not symmetric positive definite");
    EXPECT_STREQ(hlq::version, "0.1.0");
}
