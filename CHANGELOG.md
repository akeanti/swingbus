# Changelog

User-visible changes are listed here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and versions follow [semantic versioning](https://semver.org).

## [0.1.0] - 2026-10-05

First release.

### Added

- MATPOWER case reader and writer for version 1 and 2 files, including comments, cell arrays and `Inf` values.
- Fifteen bundled test systems: the IEEE 14, 30, 57, 118 and 300-bus cases from MATPOWER and ten PGLib-OPF v23.07 benchmarks.
- On-demand download of all 66 PGLib-OPF v23.07 cases, verified against pinned SHA-256 digests and cached locally.
- AC power flow with Newton-Raphson, fast-decoupled (XB and BX) and Gauss-Seidel methods, flat, DC or case initialization, and generator reactive power limits.
- DC power flow, PTDF with single or distributed slack, and LODF with islanding detection.
- N-1 contingency screening with a DC method based on LODF and an AC method that re-solves every outage.
- DC optimal power flow with quadratic and piecewise-linear costs, thermal and angle difference limits, LMPs and shadow prices, solved by a built-in primal-dual interior point method.
- Exact economic dispatch.
- DC cascading failure simulation and random scenario sampling for building graph datasets.
- Command line interface with `pf`, `opf`, `n1`, `cascade`, `cases` and `info`.
- Plotting helpers: automatic network layout, network maps, voltage profiles and convergence plots.
- Test suite that compares power flow, DC power flow, DC-OPF, admittance matrices and sensitivity factors with PYPOWER.

[0.1.0]: https://github.com/akeanti/swingbus/releases/tag/v0.1.0
