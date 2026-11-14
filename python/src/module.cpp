// Python bindings of hlq. The package wraps _Solver (float64) and _Solver32 (float32) as hlq.Solver and
// hlq.Solver32, converting array-likes on the way in.

#include <algorithm>
#include <cstddef>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <hlq/hlq.hpp>

namespace nb = nanobind;
using namespace nb::literals;

namespace {

using Index = hlq::Index;
using LevelEnds = std::vector<Index>;

/// A matrix of any strides, and a contiguous vector, read in place.
template <typename Scalar>
using InMatrix = nb::ndarray<const Scalar, nb::ndim<2>, nb::device::cpu>;
template <typename Scalar>
using InVector = nb::ndarray<const Scalar, nb::ndim<1>, nb::device::cpu, nb::c_contig>;
template <typename Scalar>
using OutVector = nb::ndarray<nb::numpy, Scalar, nb::ndim<1>>;

template <typename Scalar>
hlq::MatrixView<Scalar> view(const InMatrix<Scalar>& a) {
    return {a.data(), static_cast<Index>(a.shape(0)), static_cast<Index>(a.shape(1)),
            static_cast<Index>(a.stride(0)), static_cast<Index>(a.stride(1))};
}

template <typename Scalar>
std::span<const Scalar> view(const InVector<Scalar>& v) {
    return {v.data(), v.shape(0)};
}

/// A copy of a result as a new NumPy array, which owns its memory.
template <typename Scalar>
nb::object to_numpy(std::span<const Scalar> values) {
    auto* data = new Scalar[values.size()];
    std::copy(values.begin(), values.end(), data);
    nb::capsule owner(data, [](void* p) noexcept { delete[] static_cast<Scalar*>(p); });
    return nb::cast(OutVector<Scalar>(data, {values.size()}, owner), nb::rv_policy::move);
}

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
        .def_rw(
            "fold_final_level", &Settings::fold_final_level,
            "Take a last level of equalities with full column rank, such as x = 0, as the choice among the "
            "optima instead of searching it.")
        .def("__repr__", [](const Settings& s) {
            std::ostringstream out;
            out << "Settings(feasibility_tolerance=" << s.feasibility_tolerance
                << ", rank_tolerance=" << s.rank_tolerance << ", dual_tolerance=" << s.dual_tolerance
                << ", max_changes=" << s.max_changes
                << ", fold_final_level=" << (s.fold_final_level ? "True" : "False") << ")";
            return out.str();
        });
}

template <typename Scalar>
void bind_solver(nb::module_& m, const char* name, const char* doc) {
    using Solver = hlq::Solver<Scalar>;
    using Matrix = InMatrix<Scalar>;
    using Vector = InVector<Scalar>;

    nb::class_<Solver>(m, name, doc)
        .def(nb::init<Index, Index>(), "max_rows"_a, "cols"_a,
             "A solver for problems with up to max_rows constraints and exactly cols variables.")
        .def(
            "set_problem",
            [](Solver& s, const Matrix& matrix, const Vector& lower, const Vector& upper,
               const LevelEnds& ends) { check(s.set_problem(view(matrix), view(lower), view(upper), ends)); },
            "matrix"_a, "lower"_a, "upper"_a, "level_ends"_a,
            "Sets the problem lower <= matrix @ x <= upper, with level k being rows [level_ends[k-1], "
            "level_ends[k]). Keeps the active set as a warm start when the structure is unchanged. Raises "
            "ValueError for invalid input.")
        .def(
            "set_metric", [](Solver& s, const Matrix& metric) { check(s.set_metric(view(metric))); },
            "metric"_a,
            "Sets the symmetric positive definite metric in which the solution is closest to the reference.")
        .def(
            "set_reference",
            [](Solver& s, const Vector& reference) { check(s.set_reference(view(reference))); },
            "reference"_a, "Sets the point that the solution is closest to among the optima.")
        .def("reset", &Solver::reset, "Forgets the active set (the next solve starts cold).")
        .def("solve", &Solver::solve, nb::call_guard<nb::gil_scoped_release>(), "Solves; returns a Status.")
        .def_prop_ro(
            "solution", [](const Solver& s) { return to_numpy(s.solution()); }, "The solution.")
        .def_prop_ro(
            "violation", [](const Solver& s) { return to_numpy(s.violation()); },
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
                return to_numpy(s.multipliers(level));
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
