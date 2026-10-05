from __future__ import annotations

import warnings

import numpy as np
import pytest

import swingbus as sb
from swingbus import idx

pypower = pytest.importorskip("pypower.api")

pytestmark = pytest.mark.oracle

POWER_FLOW_CASES = [
    "case14",
    "case_ieee30",
    "case57",
    "case118",
    "case300",
    "pglib_opf_case5_pjm",
    "pglib_opf_case24_ieee_rts",
    "pglib_opf_case73_ieee_rts",
]
OPF_CASES = [
    *POWER_FLOW_CASES,
    "pglib_opf_case3_lmbd",
    "pglib_opf_case14_ieee",
    "pglib_opf_case30_ieee",
    "pglib_opf_case39_epri",
    "pglib_opf_case57_ieee",
    "pglib_opf_case118_ieee",
    "pglib_opf_case300_ieee",
]
ALGORITHMS = {"nr": 1, "fdxb": 2, "fdbx": 3, "gs": 4}


def ppc(case):
    return {
        "version": "2",
        "baseMVA": case.base_mva,
        "bus": case.bus.copy(),
        "gen": case.gen.copy(),
        "branch": case.branch.copy(),
        "gencost": None if case.gencost is None else case.gencost.copy(),
    }


def quiet(function, *args):
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        return function(*args)


def options(**extra):
    return pypower.ppoption(VERBOSE=0, OUT_ALL=0, **extra)


def per_bus(case, gen, column):
    totals = np.zeros(case.nb)
    np.add.at(totals, case.bus_index(gen[:, idx.GEN_BUS]), gen[:, column])
    return totals


@pytest.mark.parametrize("name", POWER_FLOW_CASES)
@pytest.mark.parametrize("method", list(ALGORITHMS))
def test_ac_power_flow(name, method):
    case = sb.load(name)
    if method == "gs" and case.nb > 60:
        pytest.skip("Gauss-Seidel is too slow to converge on large cases")
    ours = sb.runpf(case, method)
    theirs, success = quiet(pypower.runpf, ppc(case), options(PF_ALG=ALGORITHMS[method]))
    assert ours.converged == bool(success)
    np.testing.assert_allclose(ours.vm, theirs["bus"][:, idx.VM], atol=1e-9)
    np.testing.assert_allclose(ours.va, theirs["bus"][:, idx.VA], atol=1e-7)
    for column in (idx.PG, idx.QG):
        np.testing.assert_allclose(
            per_bus(case, ours.case.gen, column), per_bus(case, theirs["gen"], column), atol=1e-6
        )
    flows = ours.case.branch[:, idx.PF : idx.QT + 1]
    np.testing.assert_allclose(flows, theirs["branch"][:, idx.PF : idx.QT + 1], atol=1e-6)


@pytest.mark.parametrize("name", POWER_FLOW_CASES)
def test_dc_power_flow(name):
    case = sb.load(name)
    ours = sb.rundcpf(case)
    theirs, _ = quiet(pypower.rundcpf, ppc(case), options())
    np.testing.assert_allclose(ours.va, theirs["bus"][:, idx.VA], atol=1e-9)
    np.testing.assert_allclose(
        per_bus(case, ours.case.gen, idx.PG), per_bus(case, theirs["gen"], idx.PG), atol=1e-9
    )
    np.testing.assert_allclose(ours.pf, theirs["branch"][:, idx.PF], atol=1e-9)


@pytest.mark.parametrize("name", OPF_CASES)
def test_dc_optimal_power_flow(name):
    case = sb.load(name)
    ours = sb.rundcopf(case)
    theirs = quiet(pypower.rundcopf, ppc(case), options())
    assert ours.success
    assert theirs["success"]
    assert ours.objective == pytest.approx(theirs["f"], rel=1e-6)
    np.testing.assert_allclose(ours.lmp, theirs["bus"][:, idx.LAM_P], atol=1e-3)
    np.testing.assert_allclose(ours.pg, theirs["gen"][:, idx.PG], atol=1e-2)
    multipliers = slice(idx.MU_PMAX, idx.MU_PMIN + 1)
    np.testing.assert_allclose(ours.case.gen[:, multipliers], theirs["gen"][:, multipliers], atol=1e-3)
    shadow = slice(idx.MU_SF, idx.MU_ST + 1)
    np.testing.assert_allclose(ours.case.branch[:, shadow], theirs["branch"][:, shadow], atol=1e-3)


@pytest.mark.parametrize("name", ["case14", "case118", "case300", "pglib_opf_case24_ieee_rts"])
def test_admittance_and_sensitivities(name):
    from pypower.ext2int import ext2int
    from pypower.makeLODF import makeLODF
    from pypower.makePTDF import makePTDF
    from pypower.makeYbus import makeYbus

    case = sb.load(name)
    internal = quiet(ext2int, ppc(case))
    Ybus = quiet(makeYbus, internal["baseMVA"], internal["bus"], internal["branch"])[0]
    work = sb.network.to_internal(case)[0]
    np.testing.assert_allclose(sb.make_ybus(work)[0].toarray(), Ybus.toarray(), atol=1e-12)
    slack = int(np.flatnonzero(internal["bus"][:, idx.BUS_TYPE] == idx.REF)[0])
    H = quiet(makePTDF, internal["baseMVA"], internal["bus"], internal["branch"], slack)
    np.testing.assert_allclose(sb.ptdf(case), H, atol=1e-10)
    L = quiet(makeLODF, internal["branch"], H)
    ours = sb.lodf(case)
    finite = np.isfinite(ours) & np.isfinite(L) & (np.abs(L) < 1e6)
    np.testing.assert_allclose(ours[finite], L[finite], atol=1e-8)
