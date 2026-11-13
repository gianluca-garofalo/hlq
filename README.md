# hlq

[![CI](https://github.com/gianluca-garofalo/hlq/actions/workflows/ci.yml/badge.svg)](https://github.com/gianluca-garofalo/hlq/actions/workflows/ci.yml)
[![Comparisons](https://github.com/gianluca-garofalo/hlq/actions/workflows/compare.yml/badge.svg)](https://github.com/gianluca-garofalo/hlq/actions/workflows/compare.yml)

Hierarchical (lexicographic) least squares with inequality constraints, in C++20 and Python.

A problem is a stack of levels, each a set of bounded rows $l_k \le A_k x \le u_k$ (equalities where $l = u$). Level 1 minimizes the squared violation of its bounds; every following level minimizes its own among the minimizers of the levels above. Among the points that solve every level, hlq returns the one closest to a reference (zero by default) in a metric (the identity by default). This is the problem of prioritized task control in robotics, and of any multi-objective least squares with strict priorities.

hlq keeps all active rows of all levels in one Gram–Schmidt factorization taken in priority order, $A = K Q$: the solution is a forward substitution, the multipliers of every level a backward substitution, and adding or removing a constraint updates only the rows after it, by Givens rotations. See [docs/algorithm.md](docs/algorithm.md) for the derivation.

- **Exact.** Each level's least squares is solved exactly, with rank decided per level by an absolute threshold, and conflicting levels by a Householder QR rather than normal equations.
- **Real-time friendly.** `solve()` never allocates; with compile-time capacities nothing touches the heap. Setup reports invalid input with a `Status`, and nothing throws.
- **Warm starts.** Re-solving a problem with the same structure starts from the previous active set; re-solving the same problem needs no changes.
- **No dependencies.** Header-only C++20 that needs only the standard library. It reads Eigen matrices, standard containers or raw memory, in double or single precision; Python bindings.
- **Verified** with an independent optimality certificate on thousands of random problems, and against LexLS, DAQP and HierarchicalQP.

## Quick start

### C++

```cpp
#include <hlq/hlq.hpp>

#include <array>
#include <limits>

int main() {
    // Level 1: x1 <= 1. Level 2: x1 + x2 = 4. Level 3: x2 = 2 (cannot be met under level 2 with x1 <= 1).
    const std::array<double, 6> a = {1, 0,
                                     1, 1,
                                     0, 1};  // row major
    const double inf = std::numeric_limits<double>::infinity();
    const std::array<double, 3> lower = {-inf, 4, 2};
    const std::array<double, 3> upper = {1, 4, 2};

    hlq::Solver<> solver(/*max_rows=*/3, /*cols=*/2);
    // Level k is rows [level_ends[k-1], level_ends[k]).
    if (solver.set_problem(hlq::MatrixView(a.data(), 3, 2), lower, upper, {1, 2, 3}) != hlq::Status::Ready) {
        return 1;
    }
    if (solver.solve() == hlq::Status::Solved) {
        // solver.solution() is {1, 3}; solver.level_cost(2) == 1
    }
}
```

Eigen matrices and vectors can be passed as they are, as in `solver.set_problem(A, lower, upper, level_ends)`. `#include <hlq/eigen.hpp>` for `hlq::as_eigen(solver.solution())`, an Eigen view of a result.

### Python

```python
import numpy as np
import hlq

solver = hlq.Solver(max_rows=3, cols=2)
solver.set_problem([[1, 0], [1, 1], [0, 1]], lower=[-np.inf, 4, 2], upper=[1, 4, 2], level_ends=[1, 2, 3])
assert solver.solve() == hlq.Status.Solved
print(solver.solution)       # [1. 3.]
print(solver.violation)      # [0. 0. 1.]
print(solver.multipliers(2)) # multipliers of the level-3 subproblem
```

## Installing

### CMake

hlq is header-only. Either install it and use `find_package`:

```bash
cmake -S . -B build -DHLQ_BUILD_TESTS=OFF
cmake --install build --prefix /path/to/prefix
```

```cmake
find_package(hlq 0.1 REQUIRED)
target_link_libraries(your_target PRIVATE hlq::hlq)
```

or add it to your build with `FetchContent`:

```cmake
include(FetchContent)
FetchContent_Declare(hlq GIT_REPOSITORY https://github.com/gianluca-garofalo/hlq.git GIT_TAG main)
FetchContent_MakeAvailable(hlq)
target_link_libraries(your_target PRIVATE hlq::hlq)
```

The library needs nothing else. The tests, benchmarks and comparisons use Eigen 3.4: an installed one when found, otherwise (with `HLQ_FETCH_DEPENDENCIES=ON`, the default) its release tarball is downloaded.

| Option | Default | |
|---|---|---|
| `HLQ_BUILD_TESTS` | top-level | GoogleTest suite (downloads GoogleTest if needed) |
| `HLQ_BUILD_BENCHMARKS` | `OFF` | Google Benchmark timings |
| `HLQ_BUILD_COMPARISONS` | `OFF` | comparisons with LexLS, DAQP and HierarchicalQP (downloads them; needs an installed Eigen) |
| `HLQ_BUILD_PYTHON` | `OFF` | the Python module (use `pip install .` instead) |
| `HLQ_FETCH_DEPENDENCIES` | `ON` | download dependencies that are not found |
| `HLQ_WARNINGS_AS_ERRORS` | `OFF` | for hlq's own targets |

### Python

```bash
pip install .
```

builds the module with scikit-build-core and nanobind (Python 3.9 and later; one stable-ABI wheel from 3.12 on).

## Using the solver

```cpp
template <typename Scalar = double, int MaxRows = hlq::dynamic, int MaxCols = hlq::dynamic,
          int MaxLevels = hlq::dynamic>
class hlq::Solver;
```

| Call | |
|---|---|
| `Solver(max_rows, cols)` | allocates for up to `max_rows` constraints in `cols` variables (`Solver()` with compile-time capacities) |
| `set_problem(A, lower, upper, level_ends)` | copies a matrix with `rows()`, `cols()` and `A(i, j)` (Eigen, `hlq::MatrixView`) and vectors with `size()` and `v[i]` (Eigen, `std::vector`, `std::span`); returns `Status::Ready` or why the input was rejected; keeps the active set when the structure is unchanged |
| `set_metric(W)`, `set_reference(r)` | the solution minimizes $(x - r)^\top W (x - r)$ among the optima |
| `solve()` | `Status::Solved`, or `Status::MaxChanges` when the search stopped early |
| `solution()`, `violation()`, `level_cost(k)` | the point, $A x$ minus its clamp to the bounds (both `std::span`s into the solver), and each level's squared violation |
| `activity(i)`, `rank(k)`, `multipliers(k)` | the active set, the rank of each level, the multipliers of each level's subproblem |
| `settings()` | `feasibility_tolerance`, `rank_tolerance`, `dual_tolerance`, `max_changes`, `fold_final_level` |
| `reset()` | forgets the active set |

The defaults suit data of order one: tolerances of $10^{-9}$ in double and $10^{-5}$ in float. The rank tolerance is relative to the longest row of each level, and the dual tolerance to the largest multiplier of each subproblem.

A final regularization level such as $x = 0$ costs nothing. hlq takes any last level of equalities with full column rank ($x = 0$, $x = x_{\text{prev}}$, a weighted one) as its own choice among the optima instead of searching it, with the same solution, costs, ranks and multipliers; `settings().fold_final_level = false` searches it like any other level.

## Verification and performance

`benchmarks/compare.cpp` compares solutions level by level (the first level whose costs differ by more than $10^{-9}$ relatively decides) with LexLS, HierarchicalQP and DAQP. On 2000 problems per family (200 for large):

| Family | vs LexLS: better / worse | vs HierarchicalQP: better / worse, max difference | vs DAQP: better / worse |
|---|---|---|---|
| random (also with a metric, and warm from other bounds) | 0 / 0 | 0 / 0, $10^{-13}$ | 0 / 0, up to 566 within DAQP's accuracy |
| integer (unit rows) | 0 / 0 | 0 / 0, $10^{-14}$ | 3 / 0 |
| degenerate: many parallel rows (unit rows) | 0 / 0 | 0 / 0, $10^{-14}$ | 31 / 0 |
| large: 20–40 variables | 0 / 0 | 0 / 0, $10^{-13}$ | 0 / 0, 193 within DAQP's accuracy |
| random with an ill-conditioned level | 32 / 0 | 8 / 0 | 0 / 0 |
| degenerate with an ill-conditioned level | 5 / 0 | 0 / 0 | |

Cold solves from data to solution (setup included), median of 5 runs, on one core of a 2.1 GHz x86-64 VM (DAQP through its C API):

| Family | hlq | hlq without the final $x = 0$ level | HierarchicalQP | LexLS | DAQP |
|---|---|---|---|---|---|
| random (2–8 variables) | 8.2 µs | 7.7 µs | 18 µs | 21 µs | 4.5 µs |
| degenerate (3 variables) | 3.2 µs | | 5.4 µs | 7.1 µs | 2.7 µs |
| large (20–40 variables) | 83 µs | 72 µs | 662 µs | 588 µs | 56 µs |

DAQP, a dual method with softened levels, is faster; its solutions agree with the exact ones to about $10^{-6}$, and it misses the optimum on some degenerate problems. hlq is the fastest of the exact solvers: seven to eight times as fast as HierarchicalQP and LexLS on the large family. The final $x = 0$ level of these problems costs it little, since it is taken as hlq's own choice among the optima. It makes fewer active-set changes than DAQP, but each costs more: hlq keeps an orthonormal basis of the active rows, which is what makes it exact on ill-conditioned levels, where DAQP keeps the factors of their Gram matrix.

CI reruns both on every pull request (see the Comparisons workflow's summary). Known limits are listed at the end of [docs/algorithm.md](docs/algorithm.md).

## Developing

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DHLQ_WARNINGS_AS_ERRORS=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the code style and commit conventions.

## References

- A. Escande, N. Mansard and P.-B. Wieber, "Hierarchical quadratic programming: Fast online humanoid-robot motion generation", *The International Journal of Robotics Research* 33(7), 2014.
- J. W. Daniel, W. B. Gragg, L. Kaufman and G. W. Stewart, "Reorthogonalization and stable algorithms for updating the Gram–Schmidt QR factorization", *Mathematics of Computation* 30, 1976.
- D. Arnström, A. Bemporad and D. Axehill, "A dual active-set solver for embedded quadratic programming using recursive LDLᵀ updates", *IEEE Transactions on Automatic Control*, 2022.

## License

BSD 3-Clause; see [LICENSE](LICENSE).
