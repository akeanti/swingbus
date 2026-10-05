import warnings

import numpy as np
from pypower.api import ppoption, rundcopf, runpf

import swingbus as sb
from swingbus import idx

POWER_FLOW = ["case14", "case_ieee30", "case57", "case118", "case300"]
OPF = [
    "case14",
    "case118",
    "case300",
    "pglib_opf_case5_pjm",
    "pglib_opf_case24_ieee_rts",
    "pglib_opf_case39_epri",
    "pglib_opf_case73_ieee_rts",
    "pglib_opf_case118_ieee",
    "pglib_opf_case300_ieee",
]
QUIET = ppoption(VERBOSE=0, OUT_ALL=0)


def as_ppc(case):
    return {
        "version": "2",
        "baseMVA": case.base_mva,
        "bus": case.bus.copy(),
        "gen": case.gen.copy(),
        "branch": case.branch.copy(),
        "gencost": case.gencost.copy(),
    }


def gap(a, b):
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def main():
    warnings.simplefilter("ignore")
    np.seterr(all="ignore")
    print("| case | method | iterations | max dVm (pu) | max dVa (deg) | max dP flow (MW) |")
    print("|---|---|---:|---:|---:|---:|")
    for name in POWER_FLOW:
        case = sb.load(name)
        for method, algorithm in (("nr", 1), ("fdxb", 2), ("fdbx", 3)):
            ours = sb.runpf(case, method)
            theirs, _ = runpf(as_ppc(case), ppoption(QUIET, PF_ALG=algorithm))
            print(
                f"| `{name}` | {ours.method} | {ours.iterations} "
                f"| {gap(ours.vm, theirs['bus'][:, idx.VM]):.1e} "
                f"| {gap(ours.va, theirs['bus'][:, idx.VA]):.1e} "
                f"| {gap(ours.pf, theirs['branch'][:, idx.PF]):.1e} |"
            )
    print()
    print("| case | cost ($/h) | relative cost gap | max dLMP ($/MWh) | max dPg (MW) |")
    print("|---|---:|---:|---:|---:|")
    for name in OPF:
        case = sb.load(name)
        ours = sb.rundcopf(case)
        theirs = rundcopf(as_ppc(case), QUIET)
        relative = abs(ours.objective - theirs["f"]) / abs(theirs["f"])
        print(
            f"| `{name}` | {ours.objective:,.2f} | {relative:.1e} "
            f"| {gap(ours.lmp, theirs['bus'][:, idx.LAM_P]):.1e} "
            f"| {gap(ours.pg, theirs['gen'][:, idx.PG]):.1e} |"
        )


if __name__ == "__main__":
    main()
