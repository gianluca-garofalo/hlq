"""Tests of the Python module: the same problems as the C++ tests, through the Python interface."""

import doctest

import numpy as np
import pytest

import hlq

inf = np.inf


def test_docstring_example():
    failures, _ = doctest.testmod(hlq)
    assert failures == 0


def test_conflicting_level_and_dependent_row():
    solver = hlq.Solver(5, 3)
    a = [[1, 1, 0], [1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 0, 1]]
    b = [2, 2, 1, 0, 3]
    solver.set_problem(a, b, b, [1, 3, 5])
    assert solver.solve() == hlq.Status.Solved
    np.testing.assert_allclose(solver.solution, [1.5, 0.5, 3])
    assert solver.level_cost(1) == pytest.approx(0.5)
    assert solver.rank(1) == 1


def test_warm_start_and_release_two_levels_below():
    solver = hlq.Solver(3, 2)
    a = np.array([[1.0, 0], [1, 1], [0, 1]])
    solver.set_problem(a, [-inf, 4, 2], [1, 4, 2], [1, 2, 3])
    solver.solve()
    np.testing.assert_allclose(solver.solution, [1, 3])
    assert solver.activity(0) == hlq.Activity.Upper
    np.testing.assert_allclose(solver.multipliers(2), [1, -1, 1], atol=1e-12)

    solver.set_problem(a, [-inf, 4, 4], [1, 4, 4], [1, 2, 3])
    assert solver.solve() == hlq.Status.Solved
    np.testing.assert_allclose(solver.solution, [0, 4], atol=1e-12)
    assert solver.changes == 1
    assert solver.activity(0) == hlq.Activity.Inactive


def test_reference_and_metric():
    solver = hlq.Solver(1, 2)
    solver.set_problem([[1, 1]], [2], [2], [1])
    solver.solve()
    np.testing.assert_allclose(solver.solution, [1, 1])
    solver.set_reference([3, 0])
    solver.solve()
    np.testing.assert_allclose(solver.solution, [2.5, -0.5])
    solver.set_reference([0, 0])
    solver.set_metric(np.diag([1.0, 4.0]))
    solver.solve()
    np.testing.assert_allclose(solver.solution, [1.6, 0.4])


def test_settings_and_max_changes():
    solver = hlq.Solver(3, 2)
    settings = solver.settings
    assert settings.feasibility_tolerance == pytest.approx(1e-9)
    settings.max_changes = 1
    solver.settings = settings
    solver.set_problem([[1, 0], [0, 1], [1, 1]], [-inf, -inf, 4], [1, 1, 4], [2, 3])
    assert solver.solve() == hlq.Status.MaxChanges
    settings.max_changes = 500
    solver.settings = settings
    solver.reset()
    assert solver.solve() == hlq.Status.Solved
    np.testing.assert_allclose(solver.solution, [1, 1])


def test_invalid_input_raises():
    solver = hlq.Solver(3, 2)
    assert solver.solve() == hlq.Status.NoProblem
    with pytest.raises(ValueError, match="invalid level ends"):
        solver.set_problem(np.eye(2), [0, 0], [1, 1], [1, 3])
    with pytest.raises(ValueError, match="invalid bounds"):
        solver.set_problem(np.eye(2), [1, 0], [0, 1], [1, 2])
    with pytest.raises(ValueError, match="invalid dimensions"):
        solver.set_problem(np.ones((4, 2)), np.zeros(4), np.ones(4), [4])
    solver.set_problem(np.eye(2), [0, 0], [1, 1], [1, 2])
    with pytest.raises(ValueError, match="metric"):
        solver.set_metric(-np.eye(2))
    with pytest.raises(IndexError):
        solver.level_cost(5)


def test_single_precision():
    solver = hlq.Solver32(3, 2)
    solver.set_problem([[1, 0], [1, 1], [0, 1]], [-inf, 4, 2], [1, 4, 2], [1, 2, 3])
    assert solver.solve() == hlq.Status.Solved
    assert solver.solution.dtype == np.float32
    np.testing.assert_allclose(solver.solution, [1, 3], rtol=1e-5)
    assert solver.settings.feasibility_tolerance == pytest.approx(1e-5)


def test_random_problems_match_a_least_squares_check():
    # A single level of equalities: the solution is the minimum-norm least-squares solution.
    rng = np.random.default_rng(7)
    for _ in range(50):
        m, n = rng.integers(1, 8, size=2)
        a = rng.standard_normal((m, n))
        b = rng.standard_normal(m)
        solver = hlq.Solver(m, n)
        solver.set_problem(a, b, b, [m])
        assert solver.solve() == hlq.Status.Solved
        np.testing.assert_allclose(solver.solution, np.linalg.lstsq(a, b, rcond=None)[0], atol=1e-10)
