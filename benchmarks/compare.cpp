// Lexicographic comparison of hlq with LexLS, HierarchicalQP and DAQP on the random families.
//
// usage: hlq_compare [count] [family[:metric][:perturb][:ill][:unit] ...]
//
// Families: random, integer, degenerate, large. Modifiers: a random metric, a warm solve after moving the
// bounds, one ill-conditioned level, rows scaled to unit norm (DAQP minimizes the violations of unit rows, so
// it is only compared on those). Prints one line per family and exits with 1 when hlq is worse than LexLS or
// HierarchicalQP, or needs changes to re-solve a problem, on a family without :ill.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <daqp.hpp>
#include <hqp/hqp.hpp>
#include <random>
#include <string>
#include <vector>

#include <hlq/hlq.hpp>

#include "lexls_interface.hpp"
#include "support/problems.hpp"

namespace {

using hlq::test::Family;
using hlq::test::Problem;
using Index = Eigen::Index;

double level_cost(const Problem& p, const Eigen::VectorXd& x, Index k) {
    const Eigen::VectorXd ax = p.matrix.middleRows(p.begin(k), p.size(k)) * x;
    const auto lower = p.lower.segment(p.begin(k), p.size(k));
    const auto upper = p.upper.segment(p.begin(k), p.size(k));
    return ((lower - ax).cwiseMax(0.0) + (ax - upper).cwiseMax(0.0)).squaredNorm();
}

enum class Verdict { Same, Tie, FirstBetter, SecondBetter, Accuracy };

/// The first level whose costs differ by more than `relative` decides.
Verdict compare(const Problem& p, const Eigen::VectorXd& x, const Eigen::VectorXd& y, double relative) {
    if (x.isApprox(y, 1e-8) || (x - y).isZero(1e-8)) {
        return Verdict::Same;
    }
    for (Index k = 0; k < p.levels(); ++k) {
        const double cx = level_cost(p, x, k), cy = level_cost(p, y, k),
                     tol = relative * (1 + std::max(cx, cy));
        if (cx < cy - tol) {
            return Verdict::FirstBetter;
        }
        if (cy < cx - tol) {
            return Verdict::SecondBetter;
        }
    }
    return Verdict::Tie;
}

/// DAQP is accurate to about its primal tolerance: a solution within 1e-5 is "accuracy", and so is one that
/// gains at a level by giving up more than 1e-9 of violation at a level above.
Verdict compare_daqp(const Problem& p, const Eigen::VectorXd& x, const Eigen::VectorXd& d) {
    if (x.isApprox(d, 1e-8) || (x - d).isZero(1e-8)) {
        return Verdict::Same;
    }
    if ((x - d).norm() <= 1e-5 * (1 + x.norm())) {
        return Verdict::Accuracy;
    }
    bool traded = false;
    for (Index k = 0; k < p.levels(); ++k) {
        const double cx = level_cost(p, x, k), cd = level_cost(p, d, k), tol = 1e-6 * (1 + std::max(cx, cd));
        if (cx < cd - tol) {
            return Verdict::FirstBetter;
        }
        if (cd < cx - tol) {
            return traded ? Verdict::Accuracy : Verdict::SecondBetter;
        }
        traded = traded || std::sqrt(cd) > std::sqrt(cx) + 1e-9;
    }
    return Verdict::Same;
}

/// hlq against one other solver: how often hlq is lexicographically better or worse.
struct Record {
    int better = 0;
    int worse = 0;
};

struct Tally {
    Record lexls, hqp, daqp;
    int daqp_accuracy = 0, daqp_compared = 0, max_changes = 0, warm_changes = 0;
    double hqp_difference = 0;
};

Family parse_family(const std::string& name) {
    if (name == "random") {
        return Family::Random;
    }
    if (name == "integer") {
        return Family::Integer;
    }
    if (name == "degenerate") {
        return Family::Degenerate;
    }
    if (name == "large") {
        return Family::Large;
    }
    std::fprintf(stderr, "unknown family %s\n", name.c_str());
    std::exit(2);
}

Tally run(const std::string& spec, unsigned count) {
    const Family family = parse_family(spec.substr(0, spec.find(':')));
    const bool metric = spec.find(":metric") != std::string::npos;
    const bool perturb = spec.find(":perturb") != std::string::npos;
    const bool ill = spec.find(":ill") != std::string::npos;
    const bool unit = spec.find(":unit") != std::string::npos;
    Tally t;
    for (unsigned seed = 0; seed < count; ++seed) {
        Problem p = hlq::test::make_problem(family, seed);
        if (ill) {
            hlq::test::make_ill_conditioned(p, seed);
        }
        if (unit) {
            for (Index i = 0; i < p.rows(); ++i) {
                const double norm = p.matrix.row(i).norm();
                p.matrix.row(i) /= norm;
                p.lower(i) /= norm;
                p.upper(i) /= norm;
            }
        }
        hlq::Solver<> solver(p.rows(), p.cols());
        hqp::HierarchicalQP qp(p.rows(), p.cols());
        if (metric) {
            const Eigen::MatrixXd w = hlq::test::make_metric(p.cols(), seed);
            static_cast<void>(solver.set_metric(w));
            qp.set_metric(w);
        }
        static_cast<void>(solver.set_problem(p.matrix, p.lower, p.upper, p.level_ends));
        t.max_changes += solver.solve() == hlq::Status::MaxChanges ? 1 : 0;
        qp.set_problem(p.matrix, p.lower, p.upper, p.level_ends);
        static_cast<void>(qp.get_primal());
        static_cast<void>(solver.set_problem(p.matrix, p.lower, p.upper, p.level_ends));
        static_cast<void>(solver.solve());
        t.warm_changes += static_cast<int>(solver.changes());
        if (perturb) {
            std::mt19937 gen(seed + 99991u);
            std::uniform_real_distribution<double> shift(-0.3, 0.3);
            for (Index i = 0; i < p.rows() - p.cols(); ++i) {
                if (p.lower(i) == p.upper(i)) {
                    p.upper(i) = p.lower(i) += shift(gen);
                } else {
                    p.lower(i) += shift(gen);
                    p.upper(i) += shift(gen);
                    if (p.lower(i) > p.upper(i)) {
                        std::swap(p.lower(i), p.upper(i));
                    }
                }
            }
            static_cast<void>(solver.set_problem(p.matrix, p.lower, p.upper, p.level_ends));
            t.max_changes += solver.solve() == hlq::Status::MaxChanges ? 1 : 0;
            qp.set_problem(p.matrix, p.lower, p.upper, p.level_ends);
        }
        const Eigen::VectorXd x = solver.solution();
        const Eigen::VectorXd y = qp.get_primal();

        auto lexls = lexls_from_stack(p.matrix, p.upper, p.lower, p.level_ends);
        lexls.solve();
        const Eigen::VectorXd r = lexls.get_x();
        switch (compare(p, x, r, 1e-9)) {
            case Verdict::FirstBetter:
                ++t.lexls.better;
                break;
            case Verdict::SecondBetter:
                ++t.lexls.worse;
                std::printf("  %s %u: LexLS better\n", spec.c_str(), seed);
                break;
            default:
                break;
        }
        switch (compare(p, x, y, 1e-9)) {
            case Verdict::FirstBetter:
                ++t.hqp.better;
                break;
            case Verdict::SecondBetter:
                ++t.hqp.worse;
                std::printf("  %s %u: HierarchicalQP better\n", spec.c_str(), seed);
                break;
            default:
                t.hqp_difference = std::max(t.hqp_difference, (x - y).norm() / (1 + y.norm()));
                break;
        }
        if (((p.matrix.rowwise().norm().array() - 1).abs() < 1e-12).all()) {
            const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> a = p.matrix;
            Eigen::VectorXi ends(p.levels() + 1);
            ends << 0, p.level_ends;
            auto result = daqp_solve(a, p.upper, p.lower, ends);
            if (result.exitflag >= 0) {
                ++t.daqp_compared;
                switch (compare_daqp(p, x, result.get_primal())) {
                    case Verdict::FirstBetter:
                        ++t.daqp.better;
                        break;
                    case Verdict::SecondBetter:
                        ++t.daqp.worse;
                        std::printf("  %s %u: DAQP better\n", spec.c_str(), seed);
                        break;
                    case Verdict::Accuracy:
                        ++t.daqp_accuracy;
                        break;
                    default:
                        break;
                }
            }
        }
    }
    std::printf(
        "%-18s %6u | LexLS: better %4d worse %3d | HierarchicalQP: better %4d worse %3d, max |dx| %.0e | "
        "DAQP (%5d compared): better %4d worse %3d within its accuracy %4d | at max changes %d, warm changes "
        "%d\n",
        spec.c_str(), count, t.lexls.better, t.lexls.worse, t.hqp.better, t.hqp.worse, t.hqp_difference,
        t.daqp_compared, t.daqp.better, t.daqp.worse, t.daqp_accuracy, t.max_changes, t.warm_changes);
    return t;
}

}  // namespace

int main(int argc, char** argv) {
    const unsigned count = argc > 1 ? static_cast<unsigned>(std::strtoul(argv[1], nullptr, 10)) : 2000;
    std::vector<std::string> specs(argv + std::min(argc, 2), argv + argc);
    if (specs.empty()) {
        specs = {"random",  "random:metric", "random:perturb", "random:unit",
                 "integer", "integer:unit",  "degenerate",     "degenerate:unit",
                 "large",   "random:ill",    "degenerate:ill"};
    }
    std::printf(
        "hlq %s against LexLS, HierarchicalQP and DAQP: how often hlq is lexicographically better or worse\n",
        hlq::version);
    bool ok = true;
    for (const auto& spec : specs) {
        const unsigned n = spec.rfind("large", 0) == 0 ? std::max(1u, count / 10) : count;
        const Tally t = run(spec, n);
        if (spec.find(":ill") == std::string::npos) {
            ok = ok && t.lexls.worse == 0 && t.hqp.worse == 0 && t.warm_changes == 0 && t.max_changes == 0;
        }
    }
    return ok ? 0 : 1;
}
