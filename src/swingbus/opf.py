from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.sparse as sp

from swingbus import idx
from swingbus.case import Case, FloatArray, column, pad_columns
from swingbus.data import CaseLike, as_case
from swingbus.network import (
    IntArray,
    Sparse,
    bus_types,
    check_islands,
    endpoints,
    gen_buses,
    make_bdc,
    to_external,
    to_internal,
)
from swingbus.qp import solve_qp

Piece = tuple[int, FloatArray, FloatArray]


@dataclass(eq=False)
class OPFResult:
    case: Case
    success: bool
    objective: float
    iterations: int
    elapsed: float

    @property
    def lmp(self) -> FloatArray:
        return column(self.case.bus, idx.LAM_P)

    @property
    def pg(self) -> FloatArray:
        return column(self.case.gen, idx.PG)

    @property
    def va(self) -> FloatArray:
        return column(self.case.bus, idx.VA)

    @property
    def pf(self) -> FloatArray:
        return column(self.case.branch, idx.PF)

    @property
    def congestion_price(self) -> FloatArray:
        return column(self.case.branch, idx.MU_SF) + column(self.case.branch, idx.MU_ST)

    def loading(self, rating: str = "A") -> FloatArray:
        rate = column(self.case.branch, idx.RATINGS[rating.upper()])
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(rate > 0, 100.0 * np.abs(self.pf) / rate, np.nan)

    def summary(self) -> str:
        from swingbus.report import opf_summary

        return opf_summary(self)

    def to_dict(self) -> dict[str, Any]:
        from swingbus.report import opf_dict

        return opf_dict(self)

    def __str__(self) -> str:
        return self.summary()


@dataclass(eq=False)
class DispatchResult:
    pg: FloatArray
    marginal_cost: float
    cost: float
    demand: float

    def __str__(self) -> str:
        return (
            f"economic dispatch: {self.demand:.2f} MW at {self.marginal_cost:.4f} $/MWh, "
            f"cost {self.cost:.2f} $/h"
        )


def _quadratic_coefficients(gencost: FloatArray, g: int) -> FloatArray:
    n = int(gencost[g, idx.NCOST])
    if n > 3:
        raise NotImplementedError(f"cost of generator {g} has degree {n - 1}; at most 2 is supported")
    coefficients = np.zeros(3)
    coefficients[3 - n :] = gencost[g, idx.COST : idx.COST + n]
    return coefficients


def _polynomial(gencost: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
    rows = []
    for g in range(gencost.shape[0]):
        if gencost[g, idx.MODEL] != idx.POLYNOMIAL:
            raise NotImplementedError(f"economic_dispatch needs polynomial costs (generator {g})")
        rows.append(_quadratic_coefficients(gencost, g))
    table = np.array(rows, dtype=np.float64).reshape(-1, 3)
    return table[:, 0], table[:, 1], table[:, 2]


def _outputs(
    price: float, c2: FloatArray, c1: FloatArray, pmin: FloatArray, pmax: FloatArray
) -> tuple[FloatArray, FloatArray]:
    quadratic = c2 > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        ideal = np.clip((price - c1) / np.where(quadratic, 2.0 * c2, 1.0), pmin, pmax)
    low = np.where(quadratic, ideal, np.where(price > c1, pmax, pmin))
    high = np.where(quadratic, ideal, np.where(price >= c1, pmax, pmin))
    return low, high


def economic_dispatch(case: CaseLike, demand: float | None = None) -> DispatchResult:
    source = as_case(case)
    if source.gencost is None:
        raise ValueError(f"{source.name} has no generator cost data")
    on = np.flatnonzero(source.gen[:, idx.GEN_STATUS] > 0)
    c2, c1, c0 = _polynomial(source.gencost[: source.ng][on])
    pmin, pmax = source.gen[on, idx.PMIN], source.gen[on, idx.PMAX]
    if demand is None:
        energized = source.bus[:, idx.BUS_TYPE] != idx.NONE
        demand = float(source.bus[energized, idx.PD].sum())
    if not pmin.sum() - 1e-9 <= demand <= pmax.sum() + 1e-9:
        raise ValueError(
            f"demand {demand:.2f} MW is outside the committed range [{pmin.sum():.2f}, {pmax.sum():.2f}] MW"
        )
    quadratic = c2 > 0
    breakpoints = np.unique(
        np.r_[
            c1[quadratic] + 2.0 * c2[quadratic] * pmin[quadratic],
            c1[quadratic] + 2.0 * c2[quadratic] * pmax[quadratic],
            c1[~quadratic],
        ]
    )
    totals = np.array([_outputs(b, c2, c1, pmin, pmax)[1].sum() for b in breakpoints])
    k = min(int(np.searchsorted(totals, demand - 1e-9)), breakpoints.size - 1)
    low, high = _outputs(breakpoints[k], c2, c1, pmin, pmax)
    if low.sum() <= demand + 1e-9:
        price = float(breakpoints[k])
        room = high - low
        share = (demand - low.sum()) / room.sum() if room.sum() > 0 else 0.0
        output = low + share * room
    else:
        left = float(breakpoints[k - 1])
        middle = 0.5 * (left + breakpoints[k])
        ideal = (middle - c1) / np.where(quadratic, 2.0 * c2, 1.0)
        moving = quadratic & (ideal > pmin) & (ideal < pmax)
        slope = float(np.sum(1.0 / (2.0 * c2[moving])))
        price = left + (demand - _outputs(left, c2, c1, pmin, pmax)[1].sum()) / slope
        output = _outputs(price, c2, c1, pmin, pmax)[0]
    pg = np.zeros(source.ng)
    pg[on] = output
    cost = float(np.sum(c2 * output**2 + c1 * output + c0))
    return DispatchResult(pg, price, cost, float(demand))


class _Rows:
    def __init__(self, width: int) -> None:
        self.width = width
        self.blocks: list[Sparse] = []
        self.rhs: list[FloatArray] = []
        self.slices: dict[str, slice] = {}
        self.count = 0

    def add(self, name: str, block: Sparse, rhs: FloatArray) -> None:
        rows = int(block.shape[0])
        self.blocks.append(sp.csr_array(block, shape=(rows, self.width)))
        self.rhs.append(np.asarray(rhs, dtype=np.float64))
        self.slices[name] = slice(self.count, self.count + rows)
        self.count += rows

    def matrix(self) -> Sparse:
        if not self.blocks:
            return sp.csr_array((0, self.width))
        return sp.vstack(self.blocks, format="csr")

    def vector(self) -> FloatArray:
        return np.concatenate(self.rhs) if self.rhs else np.zeros(0)


def _select(columns: IntArray, width: int, sign: float = 1.0) -> Sparse:
    k = columns.size
    return sp.csr_array((np.full(k, sign), (np.arange(k), columns)), shape=(k, width))


def _difference(f: IntArray, t: IntArray, width: int) -> Sparse:
    k = f.size
    rows = np.r_[np.arange(k), np.arange(k)]
    return sp.csr_array((np.r_[np.ones(k), -np.ones(k)], (rows, np.r_[f, t])), shape=(k, width))


def _costs(
    gencost: FloatArray, base: float, nb: int, ng: int
) -> tuple[FloatArray, FloatArray, float, list[Piece]]:
    hess = np.zeros(nb + ng)
    linear = np.zeros(nb + ng)
    constant = 0.0
    pieces: list[Piece] = []
    for g in range(ng):
        model = gencost[g, idx.MODEL]
        if model == idx.POLYNOMIAL:
            c2, c1, c0 = _quadratic_coefficients(gencost, g)
            hess[nb + g] = 2.0 * c2 * base**2
            linear[nb + g] = c1 * base
            constant += c0
        elif model == idx.PW_LINEAR:
            n = int(gencost[g, idx.NCOST])
            points = gencost[g, idx.COST : idx.COST + 2 * n].reshape(n, 2)
            pieces.append((g, points[:, 0], points[:, 1]))
        else:
            raise ValueError(f"unknown cost model {model} for generator {g}")
    return hess, linear, constant, pieces


def rundcopf(case: CaseLike, *, rating: str = "A", tol: float = 1e-9, max_iter: int = 100) -> OPFResult:
    source = as_case(case)
    start = time.perf_counter()
    work, index = to_internal(source)
    if work.gencost is None:
        raise ValueError(f"{source.name} has no generator cost data")
    ref = bus_types(work)[0]
    check_islands(work, ref)
    nb, ng, nl, base = work.nb, work.ng, work.nl, work.base_mva
    Bbus, Bf, Pbusinj, Pfinj = make_bdc(work)
    hess, linear, constant, pieces = _costs(work.gencost[:ng], base, nb, ng)
    npw = len(pieces)
    width = nb + ng + npw
    hess = np.r_[hess, np.zeros(npw)]
    linear = np.r_[linear, np.ones(npw)]
    pmin = work.gen[:, idx.PMIN] / base
    pmax = work.gen[:, idx.PMAX] / base
    fixed = np.flatnonzero(np.isclose(pmin, pmax, rtol=0.0, atol=1e-10))
    free = np.setdiff1d(np.arange(ng), fixed)
    incidence = sp.csr_array((np.ones(ng), (gen_buses(work), np.arange(ng))), shape=(nb, ng))
    load = (work.bus[:, idx.PD] + work.bus[:, idx.GS]) / base

    equal = _Rows(width)
    equal.add("balance", sp.hstack([Bbus, -incidence, sp.csr_array((nb, npw))]), -load - Pbusinj)
    equal.add("reference", _select(ref, width), np.deg2rad(work.bus[ref, idx.VA]))
    equal.add("fixed", _select(nb + fixed, width), pmin[fixed])

    rate = work.branch[:, idx.RATINGS[rating.upper()]] / base
    limited = np.flatnonzero(rate > 0)
    flows = sp.hstack([Bf[limited], sp.csr_array((limited.size, ng + npw))])
    f, t = endpoints(work)
    angmin, angmax = work.branch[:, idx.ANGMIN], work.branch[:, idx.ANGMAX]
    upper = np.flatnonzero((angmax != 0) & (angmax < 360))
    lower = np.flatnonzero((angmin != 0) & (angmin > -360))

    bound = _Rows(width)
    bound.add("flow_max", flows, rate[limited] - Pfinj[limited])
    bound.add("flow_min", -flows, rate[limited] + Pfinj[limited])
    bound.add("angle_max", _difference(f[upper], t[upper], width), np.deg2rad(angmax[upper]))
    bound.add("angle_min", -_difference(f[lower], t[lower], width), -np.deg2rad(angmin[lower]))
    bound.add("pg_max", _select(nb + free, width), pmax[free])
    bound.add("pg_min", _select(nb + free, width, -1.0), -pmin[free])
    for j, (g, mw, dollars) in enumerate(pieces):
        slopes = np.diff(dollars) / np.diff(mw)
        segments = sp.hstack(
            [
                sp.csr_array((slopes.size, nb)),
                sp.csr_array(
                    (slopes * base, (np.arange(slopes.size), np.full(slopes.size, g))),
                    shape=(slopes.size, ng),
                ),
                -_select(np.full(slopes.size, j), npw),
            ]
        )
        bound.add(f"cost_{j}", segments, slopes * mw[:-1] - dollars[:-1])

    solution = solve_qp(
        sp.diags_array(hess),
        linear,
        equal.matrix(),
        equal.vector(),
        bound.matrix(),
        bound.vector(),
        tol=tol,
        max_iter=max_iter,
    )
    Va = solution.x[:nb]
    duals = solution.ineq_dual / base
    mu_flow_max, mu_flow_min = np.zeros(nl), np.zeros(nl)
    mu_flow_max[limited] = duals[bound.slices["flow_max"]]
    mu_flow_min[limited] = duals[bound.slices["flow_min"]]
    mu_pg_max, mu_pg_min = np.zeros(ng), np.zeros(ng)
    mu_pg_max[free] = duals[bound.slices["pg_max"]]
    mu_pg_min[free] = duals[bound.slices["pg_min"]]
    pinned = solution.eq_dual[equal.slices["fixed"]] / base
    mu_pg_max[fixed] = np.maximum(pinned, 0.0)
    mu_pg_min[fixed] = np.maximum(-pinned, 0.0)
    flow = (Bf @ Va + Pfinj) * base

    work.bus = pad_columns(work.bus, idx.BUS_COLS_OPF)
    work.gen = pad_columns(work.gen, idx.GEN_COLS_OPF)
    work.branch = pad_columns(work.branch, idx.BRANCH_COLS_OPF)
    work.bus[:, idx.VM] = 1.0
    work.bus[:, idx.VA] = np.rad2deg(Va)
    work.bus[:, idx.LAM_P] = solution.eq_dual[equal.slices["balance"]] / base
    work.bus[:, idx.LAM_Q] = 0.0
    work.gen[:, idx.PG] = solution.x[nb : nb + ng] * base
    work.gen[:, idx.QG] = 0.0
    work.gen[:, idx.MU_PMAX] = mu_pg_max
    work.gen[:, idx.MU_PMIN] = mu_pg_min
    work.branch[:, idx.PF] = flow
    work.branch[:, idx.PT] = -flow
    work.branch[:, idx.QF] = 0.0
    work.branch[:, idx.QT] = 0.0
    work.branch[:, idx.MU_SF] = mu_flow_max
    work.branch[:, idx.MU_ST] = mu_flow_min
    return OPFResult(
        to_external(source, work, index),
        solution.converged,
        solution.objective + constant,
        solution.iterations,
        time.perf_counter() - start,
    )
