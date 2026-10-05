from __future__ import annotations

import numpy as np
import pytest

import swingbus as sb
from swingbus import idx
from swingbus.network import (
    IslandingError,
    bridges,
    bus_types,
    check_islands,
    islands,
    to_external,
    to_internal,
)
from swingbus.solvers import JacobianPattern, jacobian


def test_ybus_is_symmetric_without_phase_shifters(case118):
    work, _ = to_internal(case118)
    Y = sb.make_ybus(work)[0].toarray()
    assert np.allclose(Y, Y.T)


def test_ybus_matches_branch_admittances(case14):
    work, _ = to_internal(case14)
    Ybus, Yf, Yt = sb.make_ybus(work)
    V = np.exp(1j * np.linspace(0, 0.3, work.nb)) * np.linspace(0.95, 1.05, work.nb)
    f = work.branch[:, idx.F_BUS].astype(int)
    t = work.branch[:, idx.T_BUS].astype(int)
    injections = np.zeros(work.nb, dtype=complex)
    np.add.at(injections, f, Yf @ V)
    np.add.at(injections, t, Yt @ V)
    shunt = (work.bus[:, idx.GS] + 1j * work.bus[:, idx.BS]) / work.base_mva * V
    np.testing.assert_allclose(Ybus @ V, injections + shunt, atol=1e-12)


def test_bdc_has_zero_row_sums(case118):
    Bbus, Bf, Pbusinj, Pfinj = sb.make_bdc(to_internal(case118)[0])
    np.testing.assert_allclose(Bbus.sum(axis=1), 0.0, atol=1e-9)
    assert Bf.shape == (case118.nl, case118.nb)
    assert np.all(Pfinj == 0)
    assert np.all(Pbusinj == 0)


@pytest.mark.parametrize("name", ["case14", "case300", "pglib_opf_case73_ieee_rts"])
def test_fast_jacobian_matches_matrix_formula(name):
    work, _ = to_internal(sb.load(name))
    Ybus = sb.make_ybus(work)[0]
    _, pv, pq = bus_types(work)
    rng = np.random.default_rng(7)
    V = (1 + 0.05 * rng.standard_normal(work.nb)) * np.exp(0.3j * rng.standard_normal(work.nb))
    fast = JacobianPattern(Ybus, pv, pq).matrix(V).toarray()
    reference = jacobian(Ybus, V, pv, pq).toarray()
    np.testing.assert_allclose(fast, reference, atol=1e-10)


def test_bridges_on_small_graphs():
    path = bridges(4, np.array([0, 1, 2]), np.array([1, 2, 3]))
    assert path.all()
    ring = bridges(4, np.array([0, 1, 2, 3]), np.array([1, 2, 3, 0]))
    assert not ring.any()
    parallel = bridges(3, np.array([0, 0, 1]), np.array([1, 1, 2]))
    assert parallel.tolist() == [False, False, True]
    self_loop = bridges(2, np.array([0, 1]), np.array([0, 1]))
    assert not self_loop.any()


def test_bridges_match_islanding_outages(case118):
    work, _ = to_internal(case118)
    f = work.branch[:, idx.F_BUS].astype(np.intp)
    t = work.branch[:, idx.T_BUS].astype(np.intp)
    flagged = bridges(work.nb, f, t)
    for k in range(work.nl):
        trial = work.copy()
        trial.branch = np.delete(trial.branch, k, axis=0)
        assert (np.unique(islands(trial)).size > 1) == flagged[k]


def test_internal_indexing_drops_offline_elements(case14):
    case = case14.copy()
    case.branch[3, idx.BR_STATUS] = 0
    case.gen[4, idx.GEN_STATUS] = 0
    case.bus[7, idx.BUS_TYPE] = idx.NONE
    work, index = to_internal(case)
    assert work.nb == 13
    assert 7 not in index.bus_rows
    assert 3 not in index.branch_rows
    assert 4 not in index.gen_rows
    assert work.bus[:, idx.BUS_I].tolist() == list(range(13))
    restored = to_external(case, work, index)
    np.testing.assert_array_equal(restored.bus[:, idx.BUS_I], case.bus[:, idx.BUS_I])


def test_reference_falls_back_to_first_pv_bus(case14):
    case = case14.copy()
    case.bus[0, idx.BUS_TYPE] = idx.PV
    ref, pv, _ = bus_types(to_internal(case)[0])
    assert ref.tolist() == [0]
    assert 0 not in pv
    case.bus[case.bus[:, idx.BUS_TYPE] == idx.PV, idx.BUS_TYPE] = idx.PQ
    with pytest.raises(ValueError, match="no reference bus"):
        bus_types(to_internal(case)[0])


def test_islands_without_reference_are_rejected(case14):
    case = case14.copy()
    case.branch[case14.branch_index(9, 14), idx.BR_STATUS] = 0
    case.branch[case14.branch_index(13, 14), idx.BR_STATUS] = 0
    work, _ = to_internal(case)
    ref = bus_types(work)[0]
    with pytest.raises(IslandingError, match="island"):
        check_islands(work, ref)
    with pytest.raises(IslandingError):
        sb.runpf(case)
