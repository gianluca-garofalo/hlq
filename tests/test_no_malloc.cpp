// solve() and the accessors after it never allocate.
//
// The target defines EIGEN_RUNTIME_NO_MALLOC: Eigen then checks every heap allocation with eigen_assert,
// which is redefined here to count failures instead of aborting, in every build type.

#include <cstdio>

namespace {
int eigen_failures = 0;
}  // namespace

#define eigen_assert(x)                                                                        \
    do {                                                                                       \
        if (!static_cast<bool>(x)) {                                                           \
            ++eigen_failures;                                                                  \
            std::fprintf(stderr, "eigen_assert failed: %s (%s:%d)\n", #x, __FILE__, __LINE__); \
        }                                                                                      \
    } while (false)

#include <gtest/gtest.h>

#include <hlq/hlq.hpp>

#include "support/problems.hpp"

namespace {

using hlq::test::Family;

/// Solves a problem with heap allocation forbidden: cold, warm, and after its bounds move.
template <typename Solver>
void solve_without_allocation(const hlq::test::Problem& p, Solver& solver) {
    ASSERT_EQ(solver.set_problem(p.matrix, p.lower, p.upper, p.level_ends), hlq::Status::Ready);
    const Eigen::VectorXd lower = p.lower.array() - 0.1;
    const Eigen::VectorXd upper = p.upper.array() + 0.1;
    Eigen::internal::set_is_malloc_allowed(false);
    EXPECT_EQ(solver.solve(), hlq::Status::Solved);
    EXPECT_EQ(solver.solve(), hlq::Status::Solved);
    for (Eigen::Index k = 0; k < p.levels(); ++k) {
        static_cast<void>(solver.multipliers(k));
        static_cast<void>(solver.level_cost(k));
    }
    // Same structure: set_problem keeps its storage and the active set.
    EXPECT_EQ(solver.set_problem(p.matrix, lower, upper, p.level_ends), hlq::Status::Ready);
    EXPECT_EQ(solver.solve(), hlq::Status::Solved);
    solver.reset();
    EXPECT_EQ(solver.solve(), hlq::Status::Solved);
    Eigen::internal::set_is_malloc_allowed(true);
}

}  // namespace

TEST(NoMalloc, DynamicSolver) {
    for (const Family family : {Family::Random, Family::Degenerate, Family::Large}) {
        for (unsigned seed = 0; seed < 200; ++seed) {
            const auto p = hlq::test::make_problem(family, seed);
            hlq::Solver<> solver(p.rows(), p.cols());
            solve_without_allocation(p, solver);
            ASSERT_EQ(eigen_failures, 0) << hlq::test::to_string(family) << " seed " << seed;
        }
    }
}

TEST(NoMalloc, FixedSizeSolver) {
    for (unsigned seed = 0; seed < 200; ++seed) {
        const auto p = hlq::test::make_problem(Family::Random, seed);
        hlq::Solver<double, 38, 8, 7> solver(p.rows(), p.cols());
        solve_without_allocation(p, solver);
        ASSERT_EQ(eigen_failures, 0) << "seed " << seed;
    }
}
