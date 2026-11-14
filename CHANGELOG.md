# Changelog

All notable changes are listed here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and versions follow [Semantic Versioning](https://semver.org/) (before 1.0, a minor version may break the interface).

## [Unreleased]

### Added

- `hlq::Solver`: hierarchical least squares with inequality constraints on one priority-ordered LQ factorization,
  with Givens updates, warm starts, a metric and a reference, and allocation-free solves.
- No dependencies beyond the C++20 standard library. The solver reads any matrix with `rows()`, `cols()` and
  `a(i, j)` (Eigen, `hlq::MatrixView`) and any vector with `size()` and `v[i]`, and returns `std::span`s;
  `hlq/eigen.hpp` maps them as Eigen vectors.
- A last level of equalities with full column rank, such as x = 0, is taken as the choice among the optima of
  the levels above instead of being searched (`Settings::fold_final_level`).
- A solve keeps the factorization of its final active set, and the next one reuses it when only the bounds or the
  reference changed.
- Compile-time capacities (`Solver<Scalar, MaxRows, MaxCols, MaxLevels>`) without heap allocation, and single
  precision.
- Python bindings (`hlq.Solver`, `hlq.Solver32`).
- Tests with an independent optimality certificate; benchmarks; comparisons with LexLS, DAQP and HierarchicalQP.
