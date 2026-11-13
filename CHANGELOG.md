# Changelog

All notable changes are listed here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and versions follow [Semantic Versioning](https://semver.org/) (before 1.0, a minor version may break the interface).

## [Unreleased]

### Added

- `hlq::Solver`: hierarchical least squares with inequality constraints on one priority-ordered LQ factorization,
  with Givens updates, warm starts, a metric and a reference, and allocation-free solves.
- Compile-time capacities (`Solver<Scalar, MaxRows, MaxCols, MaxLevels>`) without heap allocation, and single
  precision.
- Python bindings (`hlq.Solver`, `hlq.Solver32`).
- Tests with an independent optimality certificate; benchmarks; comparisons with LexLS, DAQP and HierarchicalQP.
