// solve() and the accessors after it never allocate.
//
// The global operator new of this test counts the allocations made while `counting` is set. Everything in
// that window only records its results, which are checked after it.

#include <cstddef>
#include <cstdlib>
#include <new>

namespace {

bool counting = false;
long allocations = 0;

void* allocate(std::size_t size) noexcept {
    if (counting) {
        ++allocations;
    }
    return std::malloc(size == 0 ? 1 : size);
}

}  // namespace

// Every form of operator new and delete is replaced, so that all memory comes from and returns to malloc.
void* operator new(std::size_t size) {
    if (void* p = allocate(size)) {
        return p;
    }
    throw std::bad_alloc();
}
void* operator new[](std::size_t size) {
    return operator new(size);
}
void* operator new(std::size_t size, const std::nothrow_t&) noexcept {
    return allocate(size);
}
void* operator new[](std::size_t size, const std::nothrow_t&) noexcept {
    return allocate(size);
}

// GCC sees free() on memory from operator new, not knowing that this operator new is malloc.
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmismatched-new-delete"
#endif
void operator delete(void* p) noexcept {
    std::free(p);
}
void operator delete[](void* p) noexcept {
    std::free(p);
}
void operator delete(void* p, std::size_t) noexcept {
    std::free(p);
}
void operator delete[](void* p, std::size_t) noexcept {
    std::free(p);
}
void operator delete(void* p, const std::nothrow_t&) noexcept {
    std::free(p);
}
void operator delete[](void* p, const std::nothrow_t&) noexcept {
    std::free(p);
}
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

#include <vector>

#include <gtest/gtest.h>

#include <hlq/hlq.hpp>

#include "support/problems.hpp"

namespace {

using hlq::test::Family;

/// Solves a problem with allocation counted: cold, warm, and after its bounds move. Returns the number of
/// allocations, and how many of the solves did not succeed.
template <typename Solver>
std::pair<long, int> solve_counting(const hlq::test::Problem& p, Solver& solver) {
    const Eigen::VectorXd lower = p.lower.array() - 0.1;
    const Eigen::VectorXd upper = p.upper.array() + 0.1;
    if (solver.set_problem(p.matrix, p.lower, p.upper, p.level_ends) != hlq::Status::Ready) {
        return {0, 1};
    }
    int failures = 0;
    allocations = 0;
    counting = true;
    failures += solver.solve() == hlq::Status::Solved ? 0 : 1;
    failures += solver.solve() == hlq::Status::Solved ? 0 : 1;
    for (Eigen::Index k = 0; k <= p.levels(); ++k) {
        static_cast<void>(solver.multipliers(k));
        static_cast<void>(solver.level_cost(std::min<Eigen::Index>(k, p.levels() - 1)));
    }
    static_cast<void>(solver.solution());
    static_cast<void>(solver.violation());
    // Same structure: set_problem keeps its storage and the active set.
    failures += solver.set_problem(p.matrix, lower, upper, p.level_ends) == hlq::Status::Ready ? 0 : 1;
    failures += solver.solve() == hlq::Status::Solved ? 0 : 1;
    solver.reset();
    failures += solver.solve() == hlq::Status::Solved ? 0 : 1;
    counting = false;
    return {allocations, failures};
}

}  // namespace

TEST(NoMalloc, CountsAllocations) {
    // Through a volatile, so that the compiler cannot elide the allocation.
    counting = true;
    allocations = 0;
    void* volatile probe = ::operator new(16);
    counting = false;
    ::operator delete(probe);
    EXPECT_EQ(allocations, 1);
}

TEST(NoMalloc, DynamicSolver) {
    for (const Family family : {Family::Random, Family::Degenerate, Family::Large}) {
        for (unsigned seed = 0; seed < 200; ++seed) {
            const auto p = hlq::test::make_problem(family, seed);
            hlq::Solver<> solver(p.rows(), p.cols());
            const auto [count, failures] = solve_counting(p, solver);
            ASSERT_EQ(count, 0) << hlq::test::to_string(family) << " seed " << seed;
            ASSERT_EQ(failures, 0) << hlq::test::to_string(family) << " seed " << seed;
        }
    }
}

TEST(NoMalloc, FixedSizeSolver) {
    for (unsigned seed = 0; seed < 200; ++seed) {
        const auto p = hlq::test::make_problem(Family::Random, seed);
        hlq::Solver<double, 38, 8, 7> solver(p.rows(), p.cols());
        counting = true;
        allocations = 0;
        hlq::Solver<double, 38, 8, 7> fresh(p.rows(), p.cols());
        counting = false;
        ASSERT_EQ(allocations, 0) << "a solver of fixed capacities allocated";
        const auto [count, failures] = solve_counting(p, solver);
        ASSERT_EQ(count, 0) << "seed " << seed;
        ASSERT_EQ(failures, 0) << "seed " << seed;
    }
}
