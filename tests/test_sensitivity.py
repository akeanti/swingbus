from __future__ import annotations

import numpy as np
import pytest

import swingbus as sb
from swingbus import idx
from swingbus.network import IslandingError


def injections(case):
    result = sb.rundcpf(case)
    energized = case.bus[:, idx.BUS_TYPE] != idx.NONE
    net = -case.bus[:, idx.PD] - case.bus[:, idx.GS]
    on = case.gen[:, idx.GEN_STATUS] > 0
    np.add.at(net, case.bus_index(case.gen[on, idx.GEN_BUS]), result.pg[on])
    return result, np.where(energized, net, 0.0)


@pytest.mark.parametrize("name", ["case14", "case118", "pglib_opf_case24_ieee_rts"])
def test_ptdf_reproduces_dc_flows(name):
    case = sb.load(name)
    result, net = injections(case)
    np.testing.assert_allclose(sb.ptdf(case) @ net, result.pf, atol=1e-8)


def test_slack_column_is_zero(case14):
    H = sb.ptdf(case14)
    assert np.all(H[:, 0] == 0)
    H5 = sb.ptdf(case14, slack=5)
    assert np.all(H5[:, case14.bus_index([5])[0]] == 0)
    result, net = injections(case14)
    np.testing.assert_allclose(H5 @ net, result.pf, atol=1e-8)


def test_distributed_slack(case14):
    weights = np.zeros(case14.nb)
    weights[[0, 1, 2]] = [2.0, 1.0, 1.0]
    H = sb.ptdf(case14, slack=weights)
    np.testing.assert_allclose(H @ (weights / weights.sum()), 0.0, atol=1e-12)
    with pytest.raises(ValueError, match="weights"):
        sb.ptdf(case14, slack=np.ones(3))


def test_ptdf_rejects_split_networks(case14):
    case = case14.copy()
    case.branch[case14.branch_index(9, 14), idx.BR_STATUS] = 0
    case.branch[case14.branch_index(13, 14), idx.BR_STATUS] = 0
    case.bus[case14.bus_index([14])[0], idx.BUS_TYPE] = idx.PV
    case.gen = np.vstack([case.gen, case.gen[1]])
    case.gen[-1, idx.GEN_BUS] = 14
    with pytest.raises(IslandingError):
        sb.ptdf(case)


@pytest.mark.parametrize("name", ["case14", "pglib_opf_case24_ieee_rts", "case57"])
def test_lodf_matches_resolving_each_outage(name):
    case = sb.load(name)
    L = sb.lodf(case)
    base = sb.rundcpf(case)
    for k in range(case.nl):
        if np.isnan(L[0, k]):
            continue
        trial = case.copy()
        trial.branch[k, idx.BR_STATUS] = 0
        expected = sb.rundcpf(trial).pf
        predicted = base.pf + L[:, k] * base.pf[k]
        np.testing.assert_allclose(predicted, expected, atol=1e-7)


def test_lodf_flags_radial_branches(case14):
    L = sb.lodf(case14)
    radial = np.isnan(L).any(axis=0)
    assert radial.tolist().count(True) == 1
    assert radial[case14.branch_index(7, 8)]
    assert np.all(np.diag(L)[~radial] == -1.0)


def test_lodf_ignores_offline_branches(case14):
    case = case14.copy()
    case.branch[3, idx.BR_STATUS] = 0
    L = sb.lodf(case)
    assert np.all(L[3] == 0)
    assert np.all(L[:, 3] == 0)
