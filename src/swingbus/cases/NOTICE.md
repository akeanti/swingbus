# Bundled case data

The `.m` files in this folder are test system data redistributed under their own licenses. They are not part of the swingbus source code, and the MIT license of swingbus does not cover them. Every file is byte-for-byte identical to the release it was taken from.

## IEEE test systems

`case14.m`, `case_ieee30.m`, `case57.m`, `case118.m`, `case300.m`

Power flow data from the University of Washington Power Systems Test Case Archive (https://labs.ece.uw.edu/pstca/), converted to MATPOWER format and maintained by the MATPOWER project (https://matpower.org). Taken from MATPOWER 8.1. Each file header lists the changes MATPOWER made to the original data. The PGLib-OPF distribution of the same systems records the data as Copyright (c) 1999 Richard D. Christie, University of Washington Electrical Engineering, licensed under the Creative Commons Attribution 4.0 International license (https://creativecommons.org/licenses/by/4.0/).

## PGLib-OPF benchmark systems

`pglib_opf_*.m`

From the IEEE PES Power Grid Library, Optimal Power Flow benchmarks, release v23.07 (https://github.com/power-grid-lib/pglib-opf). The data is licensed under the Creative Commons Attribution 4.0 International license. The copyright holders of each system are named in its file header. Larger PGLib-OPF systems are not bundled; `swingbus.fetch` downloads them from the same release and checks each file against a pinned SHA-256 digest.

## Citing the data

If you publish results computed on these systems, cite their sources along with swingbus:

- R. D. Christie, Power Systems Test Case Archive, University of Washington, 1999.
- S. Babaeinejadsarookolaee et al., "The Power Grid Library for Benchmarking AC Optimal Power Flow Algorithms," arXiv:1908.02788, 2019.
- R. D. Zimmerman, C. E. Murillo-Sanchez and R. J. Thomas, "MATPOWER: Steady-State Operations, Planning, and Analysis Tools for Power Systems Research and Education," IEEE Transactions on Power Systems, 26(1):12-19, 2011.
