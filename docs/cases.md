# Test cases

`swingbus.load(name)` accepts a bundled case name, any of the 66 PGLib-OPF v23.07 case names, or a path to a MATPOWER `.m` file. `swingbus cases` and `swingbus cases --pglib` print the names.

## Bundled

| name | buses | generators | branches | load (MW) | thermal limits | use it for |
|---|---:|---:|---:|---:|---|---|
| `pglib_opf_case3_lmbd` | 3 | 3 | 3 | 315 | yes | the smallest congested market |
| `pglib_opf_case5_pjm` | 5 | 5 | 6 | 1,000 | yes | the classic LMP example |
| `case14` | 14 | 5 | 20 | 259 | no | power flow homework |
| `pglib_opf_case14_ieee` | 14 | 5 | 20 | 259 | yes | IEEE 14 with ratings and costs |
| `pglib_opf_case24_ieee_rts` | 24 | 33 | 38 | 2,850 | yes | reliability studies |
| `case_ieee30` | 30 | 6 | 41 | 283 | no | power flow homework |
| `pglib_opf_case30_ieee` | 30 | 6 | 41 | 283 | yes | OPF |
| `pglib_opf_case39_epri` | 39 | 10 | 46 | 6,254 | yes | the New England system |
| `case57` | 57 | 7 | 80 | 1,251 | no | power flow |
| `pglib_opf_case57_ieee` | 57 | 7 | 80 | 1,251 | yes | OPF |
| `pglib_opf_case73_ieee_rts` | 73 | 99 | 120 | 8,550 | yes | three-area reliability system |
| `case118` | 118 | 54 | 186 | 4,242 | no | power flow, Q limits |
| `pglib_opf_case118_ieee` | 118 | 54 | 186 | 4,242 | yes | N-1, LMPs, cascades |
| `case300` | 300 | 69 | 411 | 23,526 | no | power flow on a harder case |
| `pglib_opf_case300_ieee` | 300 | 69 | 411 | 23,526 | yes | OPF |

The `case*` files are the classic IEEE systems as distributed with MATPOWER. Their bus voltages are the published solved values, which makes them good for checking a power flow by hand. They have no thermal ratings.

The `pglib_opf_*` files are the PGLib-OPF benchmark versions of the same and other systems. They add thermal ratings, angle limits and realistic generator costs, so they are the right choice for contingency analysis, prices and cascades. Their stored dispatch is an OPF starting point rather than a solved power flow; run `rundcopf` to get a consistent operating point first.

## Downloaded on demand

Every other PGLib-OPF v23.07 case, from `pglib_opf_case60_c` to `pglib_opf_case78484_epigrids`, is fetched the first time you load it:

```python
import swingbus as sb

case = sb.load("pglib_opf_case2383wp_k")
print(sb.cache_dir())
```

Downloads come from the tagged v23.07 release on GitHub and are checked against a SHA-256 digest stored in the package, so a modified file is rejected. Set `SWINGBUS_DATA` to choose another cache directory.

## Your own data

Any MATPOWER `.m` case file works: the files in the MATPOWER distribution, PGLib-OPF and other benchmark libraries, and files written by MATPOWER's `savecase` or PowerModels.jl's `export_matpower`. `swingbus.write_matpower(case, "out.m")` writes one back, and `swingbus pf case.m --save solved.m` saves a solved case from the command line.

You can also build a case in code. The tables use MATPOWER's column order, and `swingbus.idx` names every column:

```python
import swingbus as sb

case = sb.Case(
    base_mva=100,
    bus=[
        [1, 3, 0, 0, 0, 0, 1, 1.0, 0, 230, 1, 1.1, 0.9],
        [2, 1, 90, 30, 0, 0, 1, 1.0, 0, 230, 1, 1.1, 0.9],
    ],
    gen=[[1, 0, 0, 100, -100, 1.02, 100, 1, 200, 0]],
    branch=[[1, 2, 0.01, 0.1, 0.02, 150, 150, 150, 0, 0, 1, -360, 360]],
)
print(sb.runpf(case))
```

## Licenses

The bundled data is redistributed under CC BY 4.0. Copyright holders and the exact sources are listed in [`src/swingbus/cases/NOTICE.md`](../src/swingbus/cases/NOTICE.md). If you publish results on these systems, cite the data sources as well as swingbus.
