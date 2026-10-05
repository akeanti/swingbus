from __future__ import annotations

import json

import numpy as np
import pytest

import swingbus as sb
from swingbus import idx

METHODS = ["nr", "fdxb", "fdbx", "gs"]
AC_CASES = ["case14", "case_ieee30", "case57", "case118", "case300"]


def test_ieee14_reference_solution(case14):
    result = sb.runpf(case14)
    assert result.converged
    assert result.iterations == 2
    assert result.pg[0] == pytest.approx(232.39327235, abs=1e-6)
    assert result.qg[0] == pytest.approx(-16.54930054, abs=1e-6)
    assert result.vm[13] == pytest.approx(1.03552995, abs=1e-8)
    assert result.va[13] == pytest.approx(-16.03364453, abs=1e-6)
    assert result.losses.real == pytest.approx(13.39327235, abs=1e-6)
    published_vm = case14.bus[:, idx.VM]
    published_va = case14.bus[:, idx.VA]
    assert np.max(np.abs(result.vm - published_vm)) < 2e-3
    assert np.max(np.abs(result.va - published_va)) < 0.05


@pytest.mark.parametrize("name", ["case14", "case_ieee30", "case57"])
def test_all_methods_agree(name):
    case = sb.load(name)
    results = [sb.runpf(case, method) for method in METHODS]
    assert all(result.converged for result in results)
    for other in results[1:]:
        np.testing.assert_allclose(other.vm, results[0].vm, atol=1e-7)
        np.testing.assert_allclose(other.va, results[0].va, atol=1e-5)
        np.testing.assert_allclose(other.pg, results[0].pg, atol=1e-4)


@pytest.mark.parametrize("name", AC_CASES)
def test_power_balance(name):
    result = sb.runpf(sb.load(name))
    assert result.converged
    residual = result.generation - result.load - result.losses - result.shunts
    assert abs(residual) < 1e-5


@pytest.mark.parametrize("init", ["flat", "dc", "case"])
def test_initialisations_converge_to_the_same_point(case118, init):
    result = sb.runpf(case118, init=init)
    reference = sb.runpf(case118)
    assert result.converged
    np.testing.assert_allclose(result.vm, reference.vm, atol=1e-8)


def test_newton_converges_quadratically(case118):
    history = np.array(sb.runpf(case118, init="flat").history)
    tail = history[history < 1e-1]
    ratios = tail[1:] / tail[:-1] ** 2
    assert np.all(ratios < 50)
    assert history[-1] < 1e-8


def test_failure_is_reported_not_raised(case118):
    result = sb.runpf(case118, max_iter=1, init="flat")
    assert not result.converged
    assert result.iterations == 1
    assert "did not converge" in result.summary()


def test_reactive_limits_are_enforced(case118):
    free = sb.runpf(case118)
    gen = free.case.gen
    violating = (gen[:, idx.QG] > gen[:, idx.QMAX] + 1e-6) | (gen[:, idx.QG] < gen[:, idx.QMIN] - 1e-6)
    assert violating.sum() > 1
    limited = sb.runpf(case118, enforce_q_limits=True)
    assert limited.converged
    assert limited.q_limited
    gen = limited.case.gen
    ref_bus = case118.bus[case118.bus[:, idx.BUS_TYPE] == idx.REF, idx.BUS_I]
    checked = ~np.isin(gen[:, idx.GEN_BUS], ref_bus)
    assert np.all(gen[checked, idx.QG] <= gen[checked, idx.QMAX] + 1e-6)
    assert np.all(gen[checked, idx.QG] >= gen[checked, idx.QMIN] - 1e-6)
    pinned = gen[limited.q_limited]
    at_limit = np.isclose(pinned[:, idx.QG], pinned[:, idx.QMAX]) | np.isclose(
        pinned[:, idx.QG], pinned[:, idx.QMIN]
    )
    assert at_limit.all()
    np.testing.assert_array_equal(limited.case.bus[:, idx.BUS_TYPE], case118.bus[:, idx.BUS_TYPE])
    np.testing.assert_array_equal(limited.case.bus[:, idx.PD], case118.bus[:, idx.PD])


def test_dc_power_flow(case118):
    result = sb.rundcpf(case118)
    assert result.method == "DC"
    np.testing.assert_allclose(result.vm, 1.0)
    np.testing.assert_allclose(result.pf, -result.pt)
    energized = case118.bus[:, idx.BUS_TYPE] != idx.NONE
    demand = case118.bus[energized, idx.PD].sum() + case118.bus[energized, idx.GS].sum()
    assert result.pg.sum() == pytest.approx(demand, abs=1e-6)
    assert sb.runpf(case118, "dc").va[10] == result.va[10]


def test_out_of_service_branch_carries_no_flow(case14):
    case = case14.copy()
    k = case14.branch_index(2, 4)
    case.branch[k, idx.BR_STATUS] = 0
    result = sb.runpf(case)
    assert result.converged
    assert result.case.branch[k, idx.PF : idx.QT + 1].tolist() == [0.0, 0.0, 0.0, 0.0]


def test_isolated_bus_keeps_its_data(case14):
    case = case14.copy()
    row = case14.bus_index([8])[0]
    case.bus[row, idx.BUS_TYPE] = idx.NONE
    result = sb.runpf(case)
    assert result.converged
    assert result.vm[row] == case14.bus[row, idx.VM]
    assert result.case.branch[case14.branch_index(7, 8), idx.PF] == 0.0


def test_input_case_is_not_modified(case14):
    before = case14.copy()
    sb.runpf(case14, enforce_q_limits=True)
    np.testing.assert_array_equal(case14.bus, before.bus)
    np.testing.assert_array_equal(case14.gen, before.gen)
    np.testing.assert_array_equal(case14.branch, before.branch)


def test_bad_arguments(case14):
    with pytest.raises(ValueError, match="unknown method"):
        sb.runpf(case14, "magic")
    with pytest.raises(ValueError, match="unknown init"):
        sb.runpf(case14, init="random")


def test_result_views(case14):
    result = sb.runpf("case14")
    assert np.isnan(result.loading()).all()
    rated = sb.runpf("pglib_opf_case14_ieee", init="flat")
    assert np.isfinite(rated.loading()).all()
    payload = json.loads(json.dumps(result.to_dict()))
    assert payload["converged"] is True
    assert payload["bus"]["id"][13] == 14
    assert len(payload["branch"]["loading"]) == case14.nl
    report = result.report()
    assert "Vm (pu)" in report
    assert "loss (MW)" in report
    assert str(result) == result.summary()


def test_first_reference_generator_takes_the_slack():
    case = sb.load("pglib_opf_case73_ieee_rts")
    result = sb.runpf(case)
    reference = case.bus[case.bus[:, idx.BUS_TYPE] == idx.REF, idx.BUS_I]
    at_reference = np.flatnonzero(np.isin(case.gen[:, idx.GEN_BUS], reference))
    assert at_reference.size > 1
    changed = np.flatnonzero(~np.isclose(result.pg, case.gen[:, idx.PG]))
    assert changed.tolist() == [at_reference[0]]


def test_multiple_generators_share_reactive_power(case14):
    case = case14.copy()
    extra = case.gen[1].copy()
    extra[idx.PG] = 0.0
    extra[idx.QMAX], extra[idx.QMIN] = 100.0, -100.0
    case.gen = np.vstack([case.gen, extra])
    case.gencost = np.vstack([case.gencost, case.gencost[1]])
    result = sb.runpf(case)
    single = sb.runpf(case14)
    at_bus_two = result.case.gen[:, idx.GEN_BUS] == 2
    assert result.qg[at_bus_two].sum() == pytest.approx(single.qg[1], abs=1e-6)
    shares = result.qg[at_bus_two] - result.case.gen[at_bus_two, idx.QMIN]
    ranges = result.case.gen[at_bus_two, idx.QMAX] - result.case.gen[at_bus_two, idx.QMIN]
    assert shares[0] / ranges[0] == pytest.approx(shares[1] / ranges[1])
