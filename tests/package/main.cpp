// Solves a small problem through an installed hlq.

#include <cstdio>
#include <limits>

#include <hlq/hlq.hpp>

int main() {
    // Level 1: x1 <= 1. Level 2: x1 + x2 = 4. Level 3: x2 = 2.
    Eigen::Matrix<double, 3, 2> a;
    a << 1, 0, 1, 1, 0, 1;
    const double inf = std::numeric_limits<double>::infinity();
    const Eigen::Vector3d lower(-inf, 4, 2), upper(1, 4, 2);
    hlq::Solver<double, 3, 2, 3> solver;
    if (solver.set_problem(a, lower, upper, Eigen::Vector3i(1, 2, 3)) != hlq::Status::Ready
        || solver.solve() != hlq::Status::Solved) {
        return 1;
    }
    std::printf("hlq %s: x = (%g, %g)\n", hlq::version, solver.solution()(0), solver.solution()(1));
    return solver.solution().isApprox(Eigen::Vector2d(1, 3)) ? 0 : 1;
}
