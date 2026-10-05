from __future__ import annotations

import json

import numpy as np
import pytest

import swingbus as sb
from swingbus import idx


def brute_force(case, k):
    trial = case.copy()
    trial.branch[k, idx.BR_STATUS] = 0
    flows = sb.rundcpf(trial).pf
    rate = case.branch[:, idx.RATE_A]
    loading = np.where(rate > 0, 100.0 * np.abs(flows) / np.where(rate > 0, rate, 1.0), -np.inf)
    return loading.max(), int(np.argmax(loading))


def test_dc_screening_matches_brute_force(rts24):
    secure = sb.rundcopf(rts24).case
    result = sb.n1(secure)
    assert result.method == "DC"
    for position, k in enumerate(result.outages.tolist()):
        if result.islanding[position]:
            continue
        expected, worst = brute_force(secure, k)
        assert result.max_loading[position] == pytest.approx(expected, abs=1e-6)
        assert result.worst_branch[position] == worst


def test_dc_screening_reports_islanding(case14):
    result = sb.n1(case14)
    assert result.islanding.sum() == 1
    assert result.outages[result.islanding][0] == case14.branch_index(7, 8)
    assert np.isnan(result.max_loading).all()
    assert result.ranking()[-1] == np.flatnonzero(result.islanding)[0]


def test_ranking_puts_divergence_first_and_islanding_last(rts24):
    secure = sb.rundcopf(rts24).case
    result = sb.n1(secure)
    order = result.ranking()
    loading = result.max_loading[order]
    finite = loading[np.isfinite(loading)]
    assert np.all(np.diff(finite) <= 0)
    assert result.islanding[order[-1]]


def test_threshold_and_subset(rts24):
    secure = sb.rundcopf(rts24).case
    strict = sb.n1(secure, threshold=50.0, branches=[0, 1, 2, 3])
    assert strict.outages.tolist() == [0, 1, 2, 3]
    assert np.all(strict.overloads >= sb.n1(secure, branches=[0, 1, 2, 3]).overloads)
    assert strict.insecure.any()


def test_ac_screening(case14):
    result = sb.n1(case14, method="ac")
    assert result.method == "AC"
    assert result.islanding.sum() == 1
    assert result.converged[~result.islanding].all()
    assert np.all(result.vmin[~result.islanding] > 0.9)
    table = result.table(5)
    assert "Vmin (pu)" in table
    assert "islanding" in table


def test_ac_screening_flags_voltage_violations(ieee118):
    secure = sb.rundcopf(ieee118).case
    result = sb.n1(secure, method="ac", branches=[0, 1, 2, 3, 4, 5])
    assert result.converged.all()
    assert np.isfinite(result.max_loading).all()
    assert (result.voltage_violations >= 0).all()


def test_screening_output(case14):
    result = sb.n1(case14)
    payload = json.loads(json.dumps(result.to_dict()))
    assert len(payload["outages"]) == result.outages.size
    assert {"status", "max_loading", "worst_branch"} <= set(payload["outages"][0])
    assert str(result).startswith("N-1 DC screening of case14")


def test_bad_arguments(case14):
    with pytest.raises(ValueError, match="method"):
        sb.n1(case14, method="quantum")
    with pytest.raises(ValueError, match="rating"):
        sb.n1(case14, rating="Z")
