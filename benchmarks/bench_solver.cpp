// Timings of hlq: cold and warm solves, dynamic and fixed sizes, double and float.

#include <vector>

#include <benchmark/benchmark.h>

#include <hlq/hlq.hpp>

#include "support/problems.hpp"

namespace {

using hlq::test::Family;

constexpr unsigned problem_count = 200;

std::vector<hlq::test::Problem> problems(Family family) {
    std::vector<hlq::test::Problem> list;
    for (unsigned seed = 0; seed < problem_count; ++seed) {
        list.push_back(hlq::test::make_problem(family, seed));
    }
    return list;
}

/// One solver per problem, set up once; `warm` keeps the active set of a first solve.
template <typename Solver>
std::vector<Solver> prepared(const std::vector<hlq::test::Problem>& list, bool warm) {
    std::vector<Solver> solvers;
    solvers.reserve(list.size());
    for (const auto& p : list) {
        solvers.emplace_back(p.rows(), p.cols());
        static_cast<void>(solvers.back().set_problem(p.matrix, p.lower, p.upper, p.level_ends));
        if (warm) {
            static_cast<void>(solvers.back().solve());
        }
    }
    return solvers;
}

/// Cold solves (from the equalities only) of every problem of a family, in turn.
template <typename Solver>
void cold(benchmark::State& state, Family family) {
    const auto list = problems(family);
    auto solvers = prepared<Solver>(list, false);
    std::size_t i = 0;
    for (auto _ : state) {
        auto& solver = solvers[i];
        solver.reset();
        benchmark::DoNotOptimize(solver.solve());
        i = (i + 1) % solvers.size();
    }
    state.SetLabel("per problem, " + std::to_string(problem_count) + " problems");
}

/// Warm solves: the same problem again, from the active set of its last solve.
template <typename Solver>
void warm(benchmark::State& state, Family family) {
    const auto list = problems(family);
    auto solvers = prepared<Solver>(list, true);
    std::size_t i = 0;
    for (auto _ : state) {
        const auto& p = list[i];
        auto& solver = solvers[i];
        static_cast<void>(solver.set_problem(p.matrix, p.lower, p.upper, p.level_ends));
        benchmark::DoNotOptimize(solver.solve());
        i = (i + 1) % solvers.size();
    }
}

using Double = hlq::Solver<double>;
using Float = hlq::Solver<float>;
using Fixed = hlq::Solver<double, 38, 8, 7>;  // capacity of the random family

}  // namespace

BENCHMARK_CAPTURE(cold<Double>, random, Family::Random);
BENCHMARK_CAPTURE(cold<Fixed>, random_fixed_size, Family::Random);
BENCHMARK_CAPTURE(cold<Float>, random_float, Family::Random);
BENCHMARK_CAPTURE(cold<Double>, integer, Family::Integer);
BENCHMARK_CAPTURE(cold<Double>, degenerate, Family::Degenerate);
BENCHMARK_CAPTURE(cold<Double>, large, Family::Large);
BENCHMARK_CAPTURE(warm<Double>, random, Family::Random);
BENCHMARK_CAPTURE(warm<Fixed>, random_fixed_size, Family::Random);
BENCHMARK_CAPTURE(warm<Double>, large, Family::Large);
