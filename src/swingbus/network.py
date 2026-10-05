from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray
from scipy.sparse.csgraph import connected_components

from swingbus import idx
from swingbus.case import Case, FloatArray, pad_columns

IntArray = NDArray[np.intp]
ComplexArray = NDArray[np.complex128]
Sparse = Any


class IslandingError(ValueError):
    pass


@dataclass(frozen=True)
class Indexing:
    bus_rows: IntArray
    gen_rows: IntArray
    branch_rows: IntArray


def to_internal(case: Case) -> tuple[Case, Indexing]:
    bus_rows = np.flatnonzero(case.bus[:, idx.BUS_TYPE] != idx.NONE)
    numbers = case.bus[:, idx.BUS_I].astype(np.intp)
    lookup = np.full(int(np.max(numbers)) + 1, -1, dtype=np.intp)
    lookup[numbers[bus_rows]] = np.arange(bus_rows.size)
    gen_bus = lookup[case.gen[:, idx.GEN_BUS].astype(np.intp)]
    gen_rows = np.flatnonzero((case.gen[:, idx.GEN_STATUS] > 0) & (gen_bus >= 0))
    f = lookup[case.branch[:, idx.F_BUS].astype(np.intp)]
    t = lookup[case.branch[:, idx.T_BUS].astype(np.intp)]
    branch_rows = np.flatnonzero((case.branch[:, idx.BR_STATUS] > 0) & (f >= 0) & (t >= 0))
    if bus_rows.size == 0:
        raise ValueError(f"{case.name} has no energized buses")
    bus = case.bus[bus_rows].copy()
    bus[:, idx.BUS_I] = np.arange(bus_rows.size)
    gen = case.gen[gen_rows].copy()
    gen[:, idx.GEN_BUS] = gen_bus[gen_rows]
    branch = case.branch[branch_rows].copy()
    branch[:, idx.F_BUS] = f[branch_rows]
    branch[:, idx.T_BUS] = t[branch_rows]
    gencost = None
    if case.gencost is not None and case.gencost.shape[0] >= case.ng:
        gencost = case.gencost[gen_rows].copy()
    internal = Case(case.base_mva, bus, gen, branch, gencost, name=case.name)
    return internal, Indexing(bus_rows, gen_rows, branch_rows)


def to_external(source: Case, internal: Case, index: Indexing) -> Case:
    out = source.copy()
    out.bus = pad_columns(out.bus, internal.bus.shape[1])
    out.gen = pad_columns(out.gen, internal.gen.shape[1])
    out.branch = pad_columns(out.branch, internal.branch.shape[1])
    bus_cols = np.delete(np.arange(internal.bus.shape[1]), idx.BUS_I)
    gen_cols = np.delete(np.arange(internal.gen.shape[1]), idx.GEN_BUS)
    branch_cols = np.delete(np.arange(internal.branch.shape[1]), [idx.F_BUS, idx.T_BUS])
    out.bus[np.ix_(index.bus_rows, bus_cols)] = internal.bus[:, bus_cols]
    out.gen[np.ix_(index.gen_rows, gen_cols)] = internal.gen[:, gen_cols]
    out.branch[np.ix_(index.branch_rows, branch_cols)] = internal.branch[:, branch_cols]
    offline = np.setdiff1d(np.arange(out.nl), index.branch_rows)
    if offline.size and out.branch.shape[1] > idx.PF:
        out.branch[offline, idx.PF :] = 0.0
    return out


def endpoints(case: Case) -> tuple[IntArray, IntArray]:
    return case.branch[:, idx.F_BUS].astype(np.intp), case.branch[:, idx.T_BUS].astype(np.intp)


def gen_buses(case: Case) -> IntArray:
    return case.gen[:, idx.GEN_BUS].astype(np.intp)


def tap_ratios(branch: FloatArray, with_shift: bool = True) -> ComplexArray:
    tap = np.ones(branch.shape[0], dtype=np.complex128)
    nonzero = branch[:, idx.TAP] != 0
    tap[nonzero] = branch[nonzero, idx.TAP]
    if with_shift:
        tap = tap * np.exp(1j * np.deg2rad(branch[:, idx.SHIFT]))
    return tap


def make_ybus(case: Case) -> tuple[Sparse, Sparse, Sparse]:
    nb, nl = case.nb, case.nl
    branch = case.branch
    status = branch[:, idx.BR_STATUS]
    ys = status / (branch[:, idx.BR_R] + 1j * branch[:, idx.BR_X])
    bc = status * branch[:, idx.BR_B]
    tap = tap_ratios(branch)
    ytt = ys + 0.5j * bc
    yff = ytt / (tap * np.conj(tap))
    yft = -ys / np.conj(tap)
    ytf = -ys / tap
    ysh = (case.bus[:, idx.GS] + 1j * case.bus[:, idx.BS]) / case.base_mva
    f, t = endpoints(case)
    diagonal = np.arange(nb)
    ybus = sp.coo_array(
        (np.r_[yff, yft, ytf, ytt, ysh], (np.r_[f, f, t, t, diagonal], np.r_[f, t, f, t, diagonal])),
        shape=(nb, nb),
    ).tocsr()
    rows = np.r_[np.arange(nl), np.arange(nl)]
    cols = np.r_[f, t]
    yf = sp.coo_array((np.r_[yff, yft], (rows, cols)), shape=(nl, nb)).tocsr()
    yt = sp.coo_array((np.r_[ytf, ytt], (rows, cols)), shape=(nl, nb)).tocsr()
    return ybus, yf, yt


def make_sbus(case: Case) -> ComplexArray:
    on = case.gen[:, idx.GEN_STATUS] > 0
    injection = np.zeros(case.nb, dtype=np.complex128)
    np.add.at(injection, gen_buses(case)[on], case.gen[on, idx.PG] + 1j * case.gen[on, idx.QG])
    load = case.bus[:, idx.PD] + 1j * case.bus[:, idx.QD]
    return (injection - load) / case.base_mva


def make_bdc(case: Case) -> tuple[Sparse, Sparse, FloatArray, FloatArray]:
    nb, nl = case.nb, case.nl
    branch = case.branch
    b = branch[:, idx.BR_STATUS] / branch[:, idx.BR_X] / tap_ratios(branch, with_shift=False).real
    f, t = endpoints(case)
    rows = np.r_[np.arange(nl), np.arange(nl)]
    cft = sp.csr_array((np.r_[np.ones(nl), -np.ones(nl)], (rows, np.r_[f, t])), shape=(nl, nb))
    bf = sp.csr_array(sp.diags_array(b) @ cft)
    bbus = sp.csr_array(cft.T @ bf)
    pfinj = -b * np.deg2rad(branch[:, idx.SHIFT])
    pbusinj = cft.T @ pfinj
    return bbus, bf, pbusinj, pfinj


def bus_types(case: Case) -> tuple[IntArray, IntArray, IntArray]:
    has_gen = np.zeros(case.nb, dtype=bool)
    on = case.gen[:, idx.GEN_STATUS] > 0
    has_gen[gen_buses(case)[on]] = True
    kind = case.bus[:, idx.BUS_TYPE]
    ref = np.flatnonzero((kind == idx.REF) & has_gen)
    pv = np.flatnonzero((kind == idx.PV) & has_gen)
    pq = np.flatnonzero((kind == idx.PQ) | ~has_gen)
    if ref.size == 0:
        if pv.size == 0:
            raise ValueError(f"{case.name} has no reference bus and no PV bus with a generator")
        ref, pv = pv[:1], pv[1:]
    return ref, pv, pq


def islands(case: Case) -> IntArray:
    f, t = endpoints(case)
    graph = sp.csr_array((np.ones(case.nl), (f, t)), shape=(case.nb, case.nb))
    _, labels = connected_components(graph, directed=False)
    return np.asarray(labels, dtype=np.intp)


def check_islands(case: Case, ref: IntArray) -> None:
    labels = islands(case)
    orphaned = np.setdiff1d(np.unique(labels), labels[ref])
    if orphaned.size:
        buses = np.flatnonzero(np.isin(labels, orphaned))
        raise IslandingError(
            f"{buses.size} bus(es) form island(s) without a reference bus "
            f"(internal indices {buses[:10].tolist()}{' ...' if buses.size > 10 else ''})"
        )


def bridges(nb: int, f: IntArray, t: IntArray) -> NDArray[np.bool_]:
    adjacency: list[list[tuple[int, int]]] = [[] for _ in range(nb)]
    for edge, (u, v) in enumerate(zip(f.tolist(), t.tolist(), strict=True)):
        if u != v:
            adjacency[u].append((v, edge))
            adjacency[v].append((u, edge))
    order = [-1] * nb
    low = [0] * nb
    is_bridge = np.zeros(f.size, dtype=bool)
    counter = 0
    for root in range(nb):
        if order[root] >= 0:
            continue
        order[root] = low[root] = counter
        counter += 1
        stack = [(root, -1, iter(adjacency[root]))]
        while stack:
            node, via, neighbours = stack[-1]
            advanced = False
            for nxt, edge in neighbours:
                if edge == via:
                    continue
                if order[nxt] < 0:
                    order[nxt] = low[nxt] = counter
                    counter += 1
                    stack.append((nxt, edge, iter(adjacency[nxt])))
                    advanced = True
                    break
                low[node] = min(low[node], order[nxt])
            if advanced:
                continue
            stack.pop()
            if stack:
                parent = stack[-1][0]
                low[parent] = min(low[parent], low[node])
                if low[node] > order[parent]:
                    is_bridge[via] = True
    return is_bridge
