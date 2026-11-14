// Solves a small problem through an installed hlq, with the standard library only.

#include <array>
#include <cstdio>
#include <limits>

#include <hlq/hlq.hpp>

int main() {
    // Level 1: x1 <= 1. Level 2: x1 + x2 = 4. Level 3: x2 = 2.
    const std::array<double, 6> a = {1, 0, 1, 1, 0, 1};
    const double inf = std::numeric_limits<double>::infinity();
    const std::array<double, 3> lower = {-inf, 4, 2};
    const std::array<double, 3> upper = {1, 4, 2};
    hlq::Solver<double, 3, 2, 3> solver;
    if (solver.set_problem(hlq::MatrixView(a.data(), 3, 2), lower, upper, {1, 2, 3}) != hlq::Status::Ready
        || solver.solve() != hlq::Status::Solved) {
        return 1;
    }
    const auto x = solver.solution();
    std::printf("hlq %s: x = (%g, %g)\n", hlq::version, x[0], x[1]);
    return x[0] == 1 && x[1] == 3 ? 0 : 1;
}
