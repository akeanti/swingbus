# Validation and performance

Numbers from a solver are only useful if you can trust them. swingbus is checked three ways on every push.

1. **Against an independent implementation.** The `oracle` tests feed the same case data to [PYPOWER](https://github.com/rwl/PYPOWER), the Python port of MATPOWER, and compare every output: voltages, angles, generator dispatch, branch flows, prices and shadow prices. PYPOWER shares MATPOWER's formulation but none of swingbus's code.
2. **Against physics and published values.** Power balance must close, LMPs must equal the finite-difference marginal cost of load, LMPs must decompose into energy and congestion components through the PTDF, LODF predictions must equal a full re-solve of each outage, and the IEEE 14-bus solution must reproduce the published values.
3. **Against itself.** All four power flow methods must agree, and the fast Jacobian must equal the textbook matrix formulas.

Reproduce the tables below with `python benchmarks/validate.py` and `python benchmarks/timing.py`.

## Power flow against PYPOWER 5.1

Largest absolute difference over all buses and branches.

| case | method | iterations | max dVm (pu) | max dVa (deg) | max dP flow (MW) |
|---|---|---:|---:|---:|---:|
| `case14` | Newton-Raphson | 2 | 1.1e-15 | 3.2e-14 | 2.0e-13 |
| `case14` | fast-decoupled XB | 6 | 4.4e-16 | 2.0e-14 | 1.8e-13 |
| `case14` | fast-decoupled BX | 8 | 4.4e-16 | 1.1e-14 | 4.6e-13 |
| `case_ieee30` | Newton-Raphson | 2 | 8.9e-16 | 1.0e-13 | 8.0e-13 |
| `case_ieee30` | fast-decoupled XB | 7 | 2.4e-15 | 3.4e-14 | 1.1e-12 |
| `case_ieee30` | fast-decoupled BX | 8 | 2.2e-15 | 7.8e-14 | 7.1e-13 |
| `case57` | Newton-Raphson | 3 | 6.2e-15 | 8.9e-14 | 7.0e-13 |
| `case57` | fast-decoupled XB | 7 | 1.6e-15 | 1.6e-13 | 1.2e-12 |
| `case57` | fast-decoupled BX | 9 | 8.9e-16 | 3.8e-13 | 8.5e-13 |
| `case118` | Newton-Raphson | 3 | 1.0e-15 | 2.3e-13 | 2.2e-12 |
| `case118` | fast-decoupled XB | 8 | 6.7e-16 | 1.5e-13 | 3.6e-12 |
| `case118` | fast-decoupled BX | 7 | 8.9e-16 | 3.6e-13 | 6.5e-12 |
| `case300` | Newton-Raphson | 5 | 1.8e-14 | 4.5e-13 | 4.6e-11 |
| `case300` | fast-decoupled XB | 9 | 7.2e-15 | 3.6e-12 | 5.6e-11 |
| `case300` | fast-decoupled BX | 9 | 1.3e-14 | 5.6e-13 | 3.1e-11 |

The iteration counts are identical to PYPOWER's. The differences are at the level of floating point rounding.

## DC optimal power flow against PYPOWER 5.1

PYPOWER solves the same problem with its own interior point solver (PIPS), so small differences come from the two solvers stopping at slightly different points.

| case | cost ($/h) | relative cost gap | max dLMP ($/MWh) | max dPg (MW) |
|---|---:|---:|---:|---:|
| `case14` | 7,642.59 | 4.6e-09 | 2.6e-06 | 3.0e-05 |
| `case118` | 125,947.88 | 2.4e-08 | 2.2e-05 | 6.8e-04 |
| `case300` | 706,292.32 | 7.0e-14 | 5.1e-06 | 5.2e-04 |
| `pglib_opf_case5_pjm` | 17,479.90 | 4.6e-10 | 2.2e-08 | 7.5e-06 |
| `pglib_opf_case24_ieee_rts` | 61,001.24 | 6.9e-11 | 2.0e-07 | 1.8e-06 |
| `pglib_opf_case39_epri` | 136,816.16 | 3.0e-13 | 6.0e-10 | 1.3e-08 |
| `pglib_opf_case73_ieee_rts` | 183,003.72 | 6.2e-10 | 1.1e-06 | 1.6e-04 |
| `pglib_opf_case118_ieee` | 93,132.68 | 1.3e-10 | 9.6e-08 | 1.6e-06 |
| `pglib_opf_case300_ieee` | 517,585.54 | 8.0e-10 | 1.1e-06 | 2.5e-04 |

## Sensitivities

`ptdf` and `lodf` agree with PYPOWER's `makePTDF` and `makeLODF` to better than 1e-10. For radial branches PYPOWER returns very large or infinite factors; swingbus returns `NaN` and lists the outage as islanding, and the tests confirm that these are exactly the bridges of the network graph.

## Performance

Best of seven runs, measured with `benchmarks/timing.py` on a Windows 11 laptop with an AMD Ryzen CPU, Python 3.12, NumPy 2.5, SciPy 1.18. Times include building the matrices from the case data. The PEGASE and Polish cases are downloaded from PGLib-OPF on first use; Newton-Raphson starts from a flat voltage profile on them.

| case | buses | branches | NR iterations | NR (ms) | DC PF (ms) | DC-OPF (ms) |
|---|---:|---:|---:|---:|---:|---:|
| `case14` | 14 | 20 | 2 | 1.9 | 1.6 | 14 |
| `case118` | 118 | 186 | 3 | 2.9 | 1.8 | 18 |
| `case300` | 300 | 411 | 5 | 6.8 | 2.1 | 36 |
| `pglib_opf_case1354_pegase` | 1354 | 1991 | 5 | 22.9 | 5.7 | 145 |
| `pglib_opf_case2383wp_k` | 2383 | 2896 | 5 | 42.4 | 7.9 | 216 |
| `pglib_opf_case9241_pegase` | 9241 | 16049 | 7 | 266.0 | 33.0 | 1885 |

## Known limits

- The AC model is the standard balanced positive-sequence model. There are no unbalanced, three-phase or distribution-specific components.
- Only polynomial costs up to quadratic and piecewise-linear costs are supported in DC-OPF and economic dispatch.
- PTDF and LODF are dense matrices. They are practical up to a few thousand branches.
- The cascade model is a DC, quasi-steady-state model. See the [theory page](theory.md#7-dc-cascade-model) for what it leaves out.
- PGLib-OPF cases ship OPF starting points, not solved power flows, so AC power flow from their stored dispatch does not always converge (for example `pglib_opf_case39_epri`). Run `rundcopf` first and solve the power flow of that dispatch instead.
