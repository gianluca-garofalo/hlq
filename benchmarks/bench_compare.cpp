// Cold solves of the same problems by hlq, HierarchicalQP, LexLS and DAQP, from the data to the solution
// (each timing includes setting up the solver), and warm re-solves by hlq and DAQP after the bounds move.

#include <algorithm>
#include <api.h>
#include <hqp/hqp.hpp>
#include <utils.h>
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

/// The problem with its bounds moved by 1% of their size, as a control loop moves them between solves.
Problem moved(Problem p) {
    for (Eigen::Index i = 0; i < p.rows(); ++i) {
        const double shift = 0.01 * (1 + std::abs(p.lower(i)));
        p.lower(i) += std::isfinite(p.lower(i)) ? shift : 0;
        p.upper(i) += std::isfinite(p.upper(i)) ? shift : 0;
    }
    return p;
}

/// Warm re-solves: one solver per problem, set up once, then solved with the bounds moved back and forth.
void hlq_warm(benchmark::State& state, Family family) {
    const auto list = problems(family);
    std::vector<Problem> other;
    std::vector<hlq::Solver<>> solvers;
    for (const auto& p : list) {
        other.push_back(moved(p));
        solvers.emplace_back(p.rows(), p.cols());
        static_cast<void>(solvers.back().set_problem(p.matrix, p.lower, p.upper, p.level_ends));
        static_cast<void>(solvers.back().solve());
    }
    std::size_t i = 0;
    bool flip = true;
    for (auto _ : state) {
        const auto& p = flip ? other[i] : list[i];
        static_cast<void>(solvers[i].set_problem(p.matrix, p.lower, p.upper, p.level_ends));
        benchmark::DoNotOptimize(solvers[i].solve());
        if (!flip) {
            i = (i + 1) % list.size();
        }
        flip = !flip;
    }
}

/// The same for DAQP: a workspace per problem, then new bounds and a solve, which starts from its last
/// working set.
void daqp_warm(benchmark::State& state, Family family) {
    struct Warm {
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> a;
        Eigen::VectorXd lower[2], upper[2], bl, bu, primal, dual;
        Eigen::VectorXi ends;
        DAQPProblem qp;
        DAQPWorkspace work;
    };
    auto list = problems(family);
    std::vector<Warm> warm(list.size());
    for (std::size_t k = 0; k < list.size(); ++k) {
        auto& p = list[k];
        auto& w = warm[k];
        const Eigen::VectorXd norms = p.matrix.rowwise().norm();
        p.matrix.array().colwise() /= norms.array();
        p.lower.array() /= norms.array();
        p.upper.array() /= norms.array();
        const Problem q = moved(p);
        w.a = p.matrix;
        w.lower[0] = p.lower;
        w.upper[0] = p.upper;
        w.lower[1] = q.lower;
        w.upper[1] = q.upper;
        w.bl = p.lower;
        w.bu = p.upper;
        w.primal.resize(p.cols());
        w.dual.resize(p.rows());
        w.ends.resize(p.levels() + 1);
        w.ends << 0, p.level_ends;
        w.qp = {static_cast<int>(p.cols()),
                static_cast<int>(p.rows()),
                0,
                nullptr,
                nullptr,
                w.a.data(),
                w.bu.data(),
                w.bl.data(),
                nullptr,
                w.ends.data(),
                static_cast<int>(w.ends.size())};
        w.work.settings = nullptr;
        setup_daqp(&w.qp, &w.work, nullptr);
        DAQPResult result;
        result.x = w.primal.data();
        result.lam = w.dual.data();
        daqp_solve(&result, &w.work);
    }
    std::size_t i = 0;
    int side = 1;
    for (auto _ : state) {
        auto& w = warm[i];
        w.bl = w.lower[side];
        w.bu = w.upper[side];
        daqp_update_ldp(DAQP_UPDATE_d, &w.work, &w.qp);
        DAQPResult result;
        result.x = w.primal.data();
        result.lam = w.dual.data();
        daqp_solve(&result, &w.work);
        benchmark::DoNotOptimize(w.primal.data());
        if (side == 0) {
            i = (i + 1) % warm.size();
        }
        side = 1 - side;
    }
    for (auto& w : warm) {
        free_daqp_workspace(&w.work);
        free_daqp_ldp(&w.work);
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
BENCHMARK_CAPTURE(hlq_warm, random, Family::Random);
BENCHMARK_CAPTURE(daqp_warm, random, Family::Random);
BENCHMARK_CAPTURE(hlq_warm, large, Family::Large);
BENCHMARK_CAPTURE(daqp_warm, large, Family::Large);
