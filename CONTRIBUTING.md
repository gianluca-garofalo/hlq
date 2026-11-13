# Contributing

## Building and testing

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug -DHLQ_WARNINGS_AS_ERRORS=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

For the Python module, `pip install .[test]` and `python -m pytest`. For the comparisons with other solvers,
configure with `-DHLQ_BUILD_COMPARISONS=ON -DHLQ_BUILD_BENCHMARKS=ON` (they need an installed Eigen) and run
`build/benchmarks/hlq_compare`.

A change to the solver needs tests that fail without it. Random problems are checked with the optimality certificate
in `tests/support/certificate.hpp`; known solutions go in `tests/test_solver.cpp`.

## Code

- Formatting follows `.clang-format` (`clang-format -i` on the files you change); CI checks it with clang-format 18.
- Naming follows `.clang-tidy`: `CamelCase` types and template parameters, `snake_case` functions and variables,
  a trailing underscore for private members.
- Keep `solve()` free of allocation: workspace is sized by the constructor or `set_problem`, and the no-malloc tests
  check it.
- Comments say why, not what. Public interfaces carry Doxygen comments.

## Commits and pull requests

Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/): `feat:`, `fix:`, `perf:`,
`refactor:`, `test:`, `docs:`, `build:`, `ci:`, with an optional scope such as `feat(python):`. The subject is in the
imperative and lowercase, without a final period; the body explains why, wrapped at 72 columns.

Each commit builds and passes the tests on its own. Pull requests are merged by rebase, keeping a linear history, and
notable changes are added to `CHANGELOG.md`.
