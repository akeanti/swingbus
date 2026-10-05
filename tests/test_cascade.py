from __future__ import annotations

import json

import numpy as np
import pytest

import swingbus as sb
from swingbus import idx


@pytest.fixture(scope="module")
def dispatched(ieee118):
    return sb.rundcopf(ieee118).case


def test_unrated_network_does_not_cascade(case14):
    result = sb.cascade(case14, [case14.branch_index(7, 8)])
    assert not result.cascaded
    assert result.shed == pytest.approx(0.0, abs=1e-9)
    assert result.islands == 2


def test_stranded_load_is_shed(case14):
    outages = [case14.branch_index(9, 14), case14.branch_index(13, 14)]
    result = sb.cascade(case14, outages)
    assert result.shed == pytest.approx(case14.bus[13, idx.PD])
    assert result.case.bus[13, idx.BUS_TYPE] == idx.NONE
    assert result.served + result.shed == pytest.approx(result.demand)


def test_overload_cascade_invariants(dispatched):
    outages = [dispatched.branch_index(8, 5), dispatched.branch_index(26, 30)]
    result = sb.cascade(dispatched, outages)
    assert result.cascaded
    assert result.served + result.shed == pytest.approx(result.demand)
    final = result.case
    tripped = result.tripped
    assert np.all(final.branch[tripped, idx.BR_STATUS] == 0)
    on = final.branch[:, idx.BR_STATUS] > 0
    rate = final.branch[:, idx.RATE_A]
    assert np.all(np.abs(final.branch[on, idx.PF]) <= rate[on] * (1 + 1e-7))
    payload = json.loads(json.dumps(result.to_dict()))
    assert payload["stages"][0] == result.stages[0].tolist()
    assert "stage 1:" in str(result)


def test_worst_first_trips_one_branch_per_stage(dispatched):
    outages = [dispatched.branch_index(8, 5), dispatched.branch_index(26, 30)]
    result = sb.cascade(dispatched, outages, trip="worst")
    assert result.stages
    assert all(stage.size == 1 for stage in result.stages)


def test_emergency_margin_reduces_trips(dispatched):
    outages = [dispatched.branch_index(8, 5), dispatched.branch_index(26, 30)]
    tight = sb.cascade(dispatched, outages)
    loose = sb.cascade(dispatched, outages, overload=1.5)
    assert loose.tripped.size <= tight.tripped.size


def test_sampling_is_reproducible(ieee118):
    first = [r.to_dict() for r in sb.sample_cascades(ieee118, 3, k=2, seed=11)]
    second = [r.to_dict() for r in sb.sample_cascades(ieee118, 3, k=2, seed=11)]
    assert first == second
    assert {"load_scale", "dispatch", "margin", "dispatched"} <= set(first[0]["meta"])
    assert all(0.8 <= sample["meta"]["load_scale"] <= 1.2 for sample in first)


def test_sampling_arguments(case14):
    with pytest.raises(ValueError, match="outages"):
        next(sb.sample_cascades(case14, 1, k=100))
    with pytest.raises(ValueError, match="trip rule"):
        sb.cascade(case14, [0], trip="random")
