// Python bindings of hlq. The package wraps _Solver (float64) and _Solver32 (float32) as hlq.Solver and
// hlq.Solver32, converting array-likes on the way in.

#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <hlq/hlq.hpp>

namespace nb = nanobind;
using namespace nb::literals;

namespace {

using Index = Eigen::Index;
using LevelEnds = std::vector<Index>;

/// Setup errors become ValueError, with the status as the message.
void check(hlq::Status status) {
    if (status != hlq::Status::Ready) {
        throw nb::value_error(hlq::to_string(status));
    }
}

template <typename Scalar>
void bind_settings(nb::module_& m, const char* name) {
    using Settings = hlq::Settings<Scalar>;
    nb::class_<Settings>(m, name, "Tolerances and limits of a solver.")
        .def(nb::init<>())
        .def_rw("feasibility_tolerance", &Settings::feasibility_tolerance,
                "A bound is violated when exceeded by more than this.")
        .def_rw("rank_tolerance", &Settings::rank_tolerance,
                "Relative length below which a row depends on the active rows before it.")
        .def_rw("dual_tolerance", &Settings::dual_tolerance,
                "Relative magnitude below which a multiplier is zero.")
        .def_rw("max_changes", &Settings::max_changes, "Maximum active-set changes in one solve.")
        .def("__repr__", [](const Settings& s) {
            std::ostringstream out;
            out << "Settings(feasibility_tolerance=" << s.feasibility_tolerance
                << ", rank_tolerance=" << s.rank_tolerance << ", dual_tolerance=" << s.dual_tolerance
                << ", max_changes=" << s.max_changes << ")";
            return out.str();
        });
}

template <typename Scalar>
void bind_solver(nb::module_& m, const char* name, const char* doc) {
    using Solver = hlq::Solver<Scalar>;
    using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

    nb::class_<Solver>(m, name, doc)
        .def(nb::init<Index, Index>(), "max_rows"_a, "cols"_a,
             "A solver for problems with up to max_rows constraints and exactly cols variables.")
        .def(
            "set_problem",
            [](Solver& s, const Matrix& matrix, const Vector& lower, const Vector& upper,
               const LevelEnds& ends) {
                const Eigen::Map<const Eigen::Matrix<Index, Eigen::Dynamic, 1>> map(
                    ends.data(), static_cast<Index>(ends.size()));
                check(s.set_problem(matrix, lower, upper, map));
            },
            "matrix"_a, "lower"_a, "upper"_a, "level_ends"_a,
            "Sets the problem lower <= matrix @ x <= upper, with level k being rows [level_ends[k-1], "
            "level_ends[k]). Keeps the active set as a warm start when the structure is unchanged. Raises "
            "ValueError for invalid input.")
        .def(
            "set_metric", [](Solver& s, const Matrix& metric) { check(s.set_metric(metric)); }, "metric"_a,
            "Sets the symmetric positive definite metric in which the solution is closest to the reference.")
        .def(
            "set_reference", [](Solver& s, const Vector& reference) { check(s.set_reference(reference)); },
            "reference"_a, "Sets the point that the solution is closest to among the optima.")
        .def("reset", &Solver::reset, "Forgets the active set (the next solve starts cold).")
        .def("solve", &Solver::solve, nb::call_guard<nb::gil_scoped_release>(), "Solves; returns a Status.")
        .def_prop_ro(
            "solution", [](const Solver& s) { return Vector(s.solution()); }, "The solution.")
        .def_prop_ro(
            "violation", [](const Solver& s) { return Vector(s.violation()); },
            "A x minus A x clamped to the bounds, at the solution.")
        .def_prop_ro("status", &Solver::status)
        .def_prop_ro("changes", &Solver::changes, "Active-set changes in the last solve.")
        .def_prop_ro("rows", &Solver::rows)
        .def_prop_ro("cols", &Solver::cols)
        .def_prop_ro("levels", &Solver::levels)
        .def_prop_ro("max_rows", &Solver::max_rows)
        .def_prop_rw(
            "settings", [](const Solver& s) { return s.settings(); },
            [](Solver& s, const hlq::Settings<Scalar>& settings) { s.settings() = settings; },
            "A copy of the settings; assign a Settings to change them.")
        .def(
            "level_cost",
            [](const Solver& s, Index level) {
                if (level < 0 || level >= s.levels()) {
                    throw nb::index_error("level out of range");
                }
                return s.level_cost(level);
            },
            "level"_a, "Squared violation of a level at the solution.")
        .def(
            "rank",
            [](const Solver& s, Index level) {
                if (level < 0 || level >= s.levels()) {
                    throw nb::index_error("level out of range");
                }
                return s.rank(level);
            },
            "level"_a, "Independent active rows of a level at the solution.")
        .def(
            "activity",
            [](const Solver& s, Index row) {
                if (row < 0 || row >= s.rows()) {
                    throw nb::index_error("row out of range");
                }
                return s.activity(row);
            },
            "row"_a, "How a constraint stands in the active set at the solution.")
        .def(
            "multipliers",
            [](Solver& s, Index level) {
                if (level < 0 || level > s.levels()) {
                    throw nb::index_error("level out of range");
                }
                return Vector(s.multipliers(level));
            },
            "level"_a,
            "Multipliers of the subproblem of a level at the solution (level == levels: the final choice "
            "among the optima).");
}

}  // namespace

NB_MODULE(_hlq, m) {
    m.doc() =
        "Hierarchical least-squares with inequality constraints, on one priority-ordered LQ factorization.";
    m.attr("__version__") = HLQ_VERSION_INFO;

    nb::enum_<hlq::Status>(m, "Status", "Outcome of setting up or solving a problem.")
        .value("Solved", hlq::Status::Solved)
        .value("MaxChanges", hlq::Status::MaxChanges)
        .value("Ready", hlq::Status::Ready)
        .value("NoProblem", hlq::Status::NoProblem)
        .value("InvalidDimensions", hlq::Status::InvalidDimensions)
        .value("InvalidLevels", hlq::Status::InvalidLevels)
        .value("InvalidBounds", hlq::Status::InvalidBounds)
        .value("InvalidData", hlq::Status::InvalidData)
        .value("InvalidMetric", hlq::Status::InvalidMetric);
    nb::enum_<hlq::Activity>(m, "Activity", "How a constraint stands in the active set.")
        .value("Inactive", hlq::Activity::Inactive)
        .value("Lower", hlq::Activity::Lower)
        .value("Upper", hlq::Activity::Upper)
        .value("Equality", hlq::Activity::Equality);

    bind_settings<double>(m, "Settings");
    bind_settings<float>(m, "Settings32");
    bind_solver<double>(m, "_Solver", "Hierarchical least-squares solver in double precision.");
    bind_solver<float>(m, "_Solver32", "Hierarchical least-squares solver in single precision.");
}
