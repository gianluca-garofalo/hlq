"""Hierarchical least-squares with inequality constraints, on one priority-ordered LQ factorization.

A problem is a stack of levels; level k has rows A_k and bounds l_k <= A_k x <= u_k. Each level minimizes the
squared violation of its bounds among the minimizers of the levels above, and the solution is the point of the
optimal set closest to a reference (zero by default) in a metric (the identity by default).

>>> import numpy as np, hlq
>>> solver = hlq.Solver(max_rows=3, cols=2)
>>> solver.set_problem([[1, 0], [1, 1], [0, 1]], lower=[-np.inf, 4, 2], upper=[1, 4, 2], level_ends=[1, 2, 3])
>>> solver.solve()
Status.Solved
>>> solver.solution
array([1., 3.])
"""

import numpy as _np

from ._hlq import Activity, Settings, Settings32, Status, __version__
from ._hlq import _Solver, _Solver32

__all__ = ["Activity", "Settings", "Settings32", "Solver", "Solver32", "Status", "__version__"]


def _wrap(base, dtype, name, doc):
    """The compiled solver, with array-likes (lists, other dtypes) converted on the way in."""

    def set_problem(self, matrix, lower, upper, level_ends):
        matrix = _np.asarray(matrix, dtype=dtype)
        base.set_problem(self, _np.atleast_2d(matrix), _np.asarray(lower, dtype=dtype).ravel(),
                         _np.asarray(upper, dtype=dtype).ravel(), [int(e) for e in _np.ravel(level_ends)])

    def set_metric(self, metric):
        base.set_metric(self, _np.atleast_2d(_np.asarray(metric, dtype=dtype)))

    def set_reference(self, reference):
        base.set_reference(self, _np.asarray(reference, dtype=dtype).ravel())

    set_problem.__doc__ = base.set_problem.__doc__
    set_metric.__doc__ = base.set_metric.__doc__
    set_reference.__doc__ = base.set_reference.__doc__
    return type(name, (base,), {"__doc__": doc, "__module__": __name__, "set_problem": set_problem,
                                "set_metric": set_metric, "set_reference": set_reference})


Solver = _wrap(_Solver, _np.float64, "Solver", _Solver.__doc__)
Solver32 = _wrap(_Solver32, _np.float32, "Solver32", _Solver32.__doc__)
