// Cold solves of the same problems by hlq, HierarchicalQP, LexLS and DAQP, from the data to the solution
// (each timing includes setting up the solver).

#include <api.h>
#include <hqp/hqp.hpp>
#include <vector>

#include <benchmark/benchmark.h>

#include <hlq/hlq.hpp>

#include "lexls_interface.hpp"
#include "support/problems.hpp"

namespace {

using hlq::test::Family;
using hlq::test::Problem;

constexpr unsigned problem_count = 200;

std::vector<Problem> problems(Family family) {
    std::vector<Problem> list;
    for (unsigned seed = 0; seed < problem_count; ++seed) {
        list.push_back(hlq::test::make_problem(family, seed));
    }
    return list;
}

void hlq_solver(benchmark::State& state, Family family) {
    const auto list = problems(family);
    std::size_t i = 0;
    for (auto _ : state) {
        const auto& p = list[i];
        hlq::Solver<> solver(p.rows(), p.cols());
        static_cast<void>(solver.set_problem(p.matrix, p.lower, p.upper, p.level_ends));
        benchmark::DoNotOptimize(solver.solve());
        i = (i + 1) % list.size();
    }
}

/// hlq without the final level x == 0: its own choice among the optima (the minimum norm) gives the same
/// solution.
void hlq_without_regularization(benchmark::State& state, Family family) {
    auto list = problems(family);
    for (auto& p : list) {
        const Eigen::Index rows = p.rows() - p.cols();
        p = Problem{p.matrix.topRows(rows), p.lower.head(rows), p.upper.head(rows),
                    p.level_ends.head(p.levels() - 1)};
    }
    std::size_t i = 0;
    for (auto _ : state) {
        const auto& p = list[i];
        hlq::Solver<> solver(p.rows(), p.cols());
        static_cast<void>(solver.set_problem(p.matrix, p.lower, p.upper, p.level_ends));
        benchmark::DoNotOptimize(solver.solve());
        i = (i + 1) % list.size();
    }
}

void hierarchical_qp(benchmark::State& state, Family family) {
    const auto list = problems(family);
    std::size_t i = 0;
    for (auto _ : state) {
        const auto& p = list[i];
        hqp::HierarchicalQP solver(p.rows(), p.cols());
        solver.set_problem(p.matrix, p.lower, p.upper, p.level_ends);
        benchmark::DoNotOptimize(solver.get_primal().data());
        i = (i + 1) % list.size();
    }
}

void lexls(benchmark::State& state, Family family) {
    const auto list = problems(family);
    std::size_t i = 0;
    for (auto _ : state) {
        const auto& p = list[i];
        auto solver = lexls_from_stack(p.matrix, p.upper, p.lower, p.level_ends);
        solver.solve();
        benchmark::DoNotOptimize(solver.get_x().data());
        i = (i + 1) % list.size();
    }
}

void daqp(benchmark::State& state, Family family) {
    // DAQP minimizes the violations of unit rows: the rows are scaled once, outside the timing. It is called
    // through its C API, as its Eigen interface calls it, without the conversions and slacks of that
    // interface.
    auto list = problems(family);
    std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> rows;
    std::vector<Eigen::VectorXi> ends;
    for (auto& p : list) {
        const Eigen::VectorXd norms = p.matrix.rowwise().norm();
        p.matrix.array().colwise() /= norms.array();
        p.lower.array() /= norms.array();
        p.upper.array() /= norms.array();
        rows.emplace_back(p.matrix);
        ends.emplace_back(p.levels() + 1);
        ends.back() << 0, p.level_ends;
    }
    std::size_t i = 0;
    for (auto _ : state) {
        auto& p = list[i];
        std::vector<double> primal(static_cast<std::size_t>(p.cols()));
        std::vector<double> dual(static_cast<std::size_t>(p.rows()));
        DAQPResult result;
        result.x = primal.data();
        result.lam = dual.data();
        DAQPProblem qp = {static_cast<int>(p.cols()),
                          static_cast<int>(p.rows()),
                          0,
                          nullptr,
                          nullptr,
                          rows[i].data(),
                          p.upper.data(),
                          p.lower.data(),
                          nullptr,
                          ends[i].data(),
                          static_cast<int>(ends[i].size())};
        daqp_quadprog(&result, &qp, nullptr);
        benchmark::DoNotOptimize(primal.data());
        i = (i + 1) % list.size();
    }
}

}  // namespace

BENCHMARK_CAPTURE(hlq_solver, random, Family::Random);
BENCHMARK_CAPTURE(hlq_without_regularization, random, Family::Random);
BENCHMARK_CAPTURE(hierarchical_qp, random, Family::Random);
BENCHMARK_CAPTURE(lexls, random, Family::Random);
BENCHMARK_CAPTURE(daqp, random, Family::Random);
BENCHMARK_CAPTURE(hlq_solver, degenerate, Family::Degenerate);
BENCHMARK_CAPTURE(hierarchical_qp, degenerate, Family::Degenerate);
BENCHMARK_CAPTURE(lexls, degenerate, Family::Degenerate);
BENCHMARK_CAPTURE(daqp, degenerate, Family::Degenerate);
BENCHMARK_CAPTURE(hlq_solver, large, Family::Large);
BENCHMARK_CAPTURE(hlq_without_regularization, large, Family::Large);
BENCHMARK_CAPTURE(hierarchical_qp, large, Family::Large);
BENCHMARK_CAPTURE(lexls, large, Family::Large);
BENCHMARK_CAPTURE(daqp, large, Family::Large);
