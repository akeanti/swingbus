from __future__ import annotations

import json

import numpy as np
import pytest

import swingbus as sb
from swingbus import idx


def test_pjm_five_bus_prices(pjm5):
    result = sb.rundcopf(pjm5)
    assert result.success
    assert result.objective == pytest.approx(17479.8969, abs=1e-3)
    lmp = dict(zip(pjm5.bus[:, idx.BUS_I].astype(int).tolist(), result.lmp.tolist(), strict=True))
    assert lmp[5] == pytest.approx(10.0, abs=1e-6)
    assert lmp[4] == pytest.approx(39.9427, abs=1e-3)
    congested = np.flatnonzero(result.congestion_price > 1e-6)
    assert [pjm5.branch[k, idx.F_BUS : idx.T_BUS + 1].astype(int).tolist() for k in congested] == [[4, 5]]
    k = int(congested[0])
    assert abs(result.pf[k]) == pytest.approx(pjm5.branch[k, idx.RATE_A], abs=1e-6)


@pytest.mark.parametrize("name", ["case_ieee30", "pglib_opf_case24_ieee_rts"])
def test_lmp_equals_marginal_cost_of_load(name):
    case = sb.load(name)
    base = sb.rundcopf(case, tol=1e-11)
    step = 1e-3
    rows = np.flatnonzero(case.bus[:, idx.PD] > 0)[:4]
    for row in rows.tolist():
        bumped = case.copy()
        bumped.bus[row, idx.PD] += step
        marginal = (sb.rundcopf(bumped, tol=1e-11).objective - base.objective) / step
        assert marginal == pytest.approx(base.lmp[row], rel=2e-3, abs=2e-3)


@pytest.mark.parametrize("name", ["pglib_opf_case5_pjm", "pglib_opf_case118_ieee", "pglib_opf_case300_ieee"])
def test_lmp_decomposes_into_energy_and_congestion(name):
    case = sb.load(name)
    case.branch[:, idx.ANGMIN], case.branch[:, idx.ANGMAX] = -360, 360
    result = sb.rundcopf(case, tol=1e-11)
    reference = int(np.flatnonzero(case.bus[:, idx.BUS_TYPE] == idx.REF)[0])
    shadow = result.case.branch[:, idx.MU_SF] - result.case.branch[:, idx.MU_ST]
    rebuilt = result.lmp[reference] - sb.ptdf(case).T @ shadow
    np.testing.assert_allclose(rebuilt, result.lmp, atol=1e-8)


def test_uncongested_network_has_a_single_price(case14):
    case = case14.copy()
    case.branch[:, idx.RATE_A] = 0
    case.branch[:, idx.ANGMIN], case.branch[:, idx.ANGMAX] = -360, 360
    result = sb.rundcopf(case)
    dispatch = sb.economic_dispatch(case)
    np.testing.assert_allclose(result.lmp, dispatch.marginal_cost, rtol=1e-6)
    np.testing.assert_allclose(result.pg, dispatch.pg, atol=1e-3)
    assert result.objective == pytest.approx(dispatch.cost, rel=1e-8)


def test_economic_dispatch_merit_order(pjm5):
    dispatch = sb.economic_dispatch(pjm5, demand=600.0)
    assert dispatch.pg.sum() == pytest.approx(600.0)
    assert dispatch.marginal_cost == pytest.approx(10.0)
    assert dispatch.pg[4] == pytest.approx(600.0)
    expensive = sb.economic_dispatch(pjm5, demand=1200.0)
    assert expensive.pg.sum() == pytest.approx(1200.0)
    assert expensive.marginal_cost == pytest.approx(30.0)
    assert "economic dispatch" in str(expensive)
    with pytest.raises(ValueError, match="outside"):
        sb.economic_dispatch(pjm5, demand=10_000.0)


def test_economic_dispatch_equal_incremental_cost(case14):
    dispatch = sb.economic_dispatch(case14)
    cost = case14.gencost[: case14.ng]
    marginal = 2 * cost[:, idx.COST] * dispatch.pg + cost[:, idx.COST + 1]
    inside = (dispatch.pg > case14.gen[:, idx.PMIN] + 1e-6) & (dispatch.pg < case14.gen[:, idx.PMAX] - 1e-6)
    np.testing.assert_allclose(marginal[inside], dispatch.marginal_cost, rtol=1e-9)
    assert dispatch.pg.sum() == pytest.approx(case14.bus[:, idx.PD].sum())


def test_piecewise_linear_costs_approach_the_quadratic_optimum(case14):
    quadratic = sb.rundcopf(case14)
    case = case14.copy()
    points = 40
    table = np.zeros((case.ng, 4 + 2 * points))
    for g in range(case.ng):
        c2, c1, c0 = case14.gencost[g, idx.COST : idx.COST + 3]
        mw = np.linspace(case.gen[g, idx.PMIN], case.gen[g, idx.PMAX], points)
        table[g, :4] = [idx.PW_LINEAR, 0, 0, points]
        table[g, 4::2] = mw
        table[g, 5::2] = c2 * mw**2 + c1 * mw + c0
    case.gencost = table
    linear = sb.rundcopf(case)
    assert linear.success
    assert linear.objective == pytest.approx(quadratic.objective, rel=2e-3)
    assert linear.objective >= quadratic.objective - 1e-6


def test_fixed_generators_and_missing_costs(pjm5):
    case = pjm5.copy()
    case.gen[0, idx.PMIN] = case.gen[0, idx.PMAX] = 20.0
    result = sb.rundcopf(case)
    assert result.pg[0] == pytest.approx(20.0, abs=1e-8)
    assert result.case.gen[0, idx.MU_PMAX] > 0
    case.gencost = None
    with pytest.raises(ValueError, match="cost"):
        sb.rundcopf(case)
    with pytest.raises(ValueError, match="cost"):
        sb.economic_dispatch(case)


def test_unsupported_costs(case14):
    case = case14.copy()
    case.gencost = np.hstack([case.gencost[:, :4], np.ones((case.ng, 4))])
    case.gencost[:, idx.NCOST] = 4
    with pytest.raises(NotImplementedError):
        sb.rundcopf(case)
    case.gencost[:, idx.MODEL] = 7
    case.gencost[:, idx.NCOST] = 3
    with pytest.raises(ValueError, match="cost model"):
        sb.rundcopf(case)


def test_opf_output(pjm5):
    result = sb.rundcopf(pjm5)
    payload = json.loads(json.dumps(result.to_dict()))
    assert payload["success"] is True
    assert len(payload["bus"]["lmp"]) == pjm5.nb
    text = str(result)
    assert "congested branch" in text
    assert "4-5" in text
    assert np.nanmax(result.loading()) == pytest.approx(100.0, abs=1e-6)
