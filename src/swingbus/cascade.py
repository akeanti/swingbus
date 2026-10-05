from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from swingbus import idx
from swingbus.case import Case, FloatArray
from swingbus.data import CaseLike, as_case
from swingbus.network import IntArray, islands, to_internal
from swingbus.opf import rundcopf
from swingbus.powerflow import rundcpf

BALANCE_TOLERANCE = 1e-9


@dataclass(eq=False)
class CascadeResult:
    case: Case
    initial: IntArray
    stages: list[IntArray]
    demand: float
    served: float
    islands: int
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def tripped(self) -> IntArray:
        if not self.stages:
            return np.zeros(0, dtype=np.intp)
        return np.concatenate(self.stages)

    @property
    def shed(self) -> float:
        return max(0.0, self.demand - self.served)

    @property
    def cascaded(self) -> bool:
        return bool(self.tripped.size)

    def summary(self) -> str:
        from swingbus.report import cascade_summary

        return cascade_summary(self)

    def to_dict(self) -> dict[str, Any]:
        from swingbus.report import cascade_dict

        return cascade_dict(self)

    def __str__(self) -> str:
        return self.summary()


def _redispatch(pg: FloatArray, pmin: FloatArray, pmax: FloatArray, target: float) -> FloatArray:
    pg = np.clip(pg, pmin, pmax)
    weights = np.maximum(pmax, 0.0)
    if weights.sum() <= 0:
        weights = np.ones_like(pmax)
    for _ in range(pg.size + 1):
        gap = target - pg.sum()
        if abs(gap) <= BALANCE_TOLERANCE:
            return pg
        movable = pg < pmax - BALANCE_TOLERANCE if gap > 0 else pg > pmin + BALANCE_TOLERANCE
        if not movable.any() or weights[movable].sum() <= 0:
            break
        pg[movable] += gap * weights[movable] / weights[movable].sum()
        pg = np.clip(pg, pmin, pmax)
    total = pg.sum()
    if target < total and total > 0:
        pg = pg * max(target, 0.0) / total
    return pg


def rebalance(state: Case) -> None:
    work, index = to_internal(state)
    labels = islands(work)
    gen_bus = work.gen[:, idx.GEN_BUS].astype(np.intp)
    for label in np.unique(labels).tolist():
        members = np.flatnonzero(labels == label)
        rows = index.bus_rows[members]
        units = index.gen_rows[np.isin(gen_bus, members)]
        bus = state.bus
        pmax = state.gen[units, idx.PMAX]
        pmin = state.gen[units, idx.PMIN]
        if units.size == 0 or pmax.sum() <= 0:
            bus[rows, idx.PD] = 0.0
            bus[rows, idx.QD] = 0.0
            bus[rows, idx.BUS_TYPE] = idx.NONE
            continue
        demand = bus[rows, idx.PD].sum() + bus[rows, idx.GS].sum()
        excess = demand - pmax.sum()
        positive = rows[bus[rows, idx.PD] > 0]
        if excess > BALANCE_TOLERANCE and positive.size:
            factor = max(0.0, 1.0 - excess / bus[positive, idx.PD].sum())
            bus[positive, idx.PD] *= factor
            bus[positive, idx.QD] *= factor
            demand = bus[rows, idx.PD].sum() + bus[rows, idx.GS].sum()
        state.gen[units, idx.PG] = _redispatch(state.gen[units, idx.PG], pmin, pmax, demand)
        anchors = rows[bus[rows, idx.BUS_TYPE] == idx.REF]
        bus[anchors, idx.BUS_TYPE] = idx.PV
        biggest = units[int(np.argmax(pmax))]
        slack = state.bus_index(state.gen[biggest, idx.GEN_BUS])[0]
        bus[slack, idx.BUS_TYPE] = idx.REF


def _served(state: Case) -> float:
    energized = state.bus[:, idx.BUS_TYPE] != idx.NONE
    return float(state.bus[energized, idx.PD].sum())


def cascade(
    case: CaseLike,
    outages: ArrayLike,
    *,
    rating: str = "A",
    overload: float = 1.0,
    trip: str = "all",
    max_stages: int = 100,
) -> CascadeResult:
    source = as_case(case)
    if trip not in {"all", "worst"}:
        raise ValueError(f"unknown trip rule {trip!r}, expected 'all' or 'worst'")
    column = idx.RATINGS[rating.upper()]
    state = source.copy()
    initial = np.atleast_1d(np.asarray(outages, dtype=np.intp))
    state.branch[initial, idx.BR_STATUS] = 0
    demand = _served(source)
    stages: list[IntArray] = []
    for _ in range(max_stages + 1):
        rebalance(state)
        result = rundcpf(state)
        state = result.case
        rate = state.branch[:, column]
        on = state.branch[:, idx.BR_STATUS] > 0
        with np.errstate(divide="ignore", invalid="ignore"):
            loading = np.where(on & (rate > 0), np.abs(state.branch[:, idx.PF]) / rate, 0.0)
        over = np.flatnonzero(loading > overload + BALANCE_TOLERANCE)
        if over.size == 0 or len(stages) == max_stages:
            break
        if trip == "worst":
            over = over[[int(np.argmax(loading[over]))]]
        state.branch[over, idx.BR_STATUS] = 0
        stages.append(over)
    work, _ = to_internal(state)
    return CascadeResult(
        case=state,
        initial=initial,
        stages=stages,
        demand=demand,
        served=_served(state),
        islands=int(np.unique(islands(work)).size),
    )


def sample_cascades(
    case: CaseLike,
    count: int,
    *,
    k: int = 2,
    load_range: tuple[float, float] = (0.8, 1.2),
    dispatch: str = "opf",
    margin: float = 0.9,
    rating: str = "A",
    overload: float = 1.0,
    seed: int | None = None,
) -> Iterator[CascadeResult]:
    source = as_case(case)
    if dispatch not in {"opf", "case"}:
        raise ValueError(f"unknown dispatch {dispatch!r}, expected 'opf' or 'case'")
    column = idx.RATINGS[rating.upper()]
    rng = np.random.default_rng(seed)
    candidates = to_internal(source)[1].branch_rows
    if k > candidates.size:
        raise ValueError(f"cannot draw {k} outages from {candidates.size} in-service branches")
    for _ in range(count):
        scale = float(rng.uniform(*load_range))
        scenario = source.copy()
        scenario.bus[:, idx.PD] *= scale
        scenario.bus[:, idx.QD] *= scale
        planned_ok = False
        if dispatch == "opf":
            tightened = scenario.copy()
            tightened.branch[:, column] *= margin
            planned = rundcopf(tightened, rating=rating)
            if planned.success:
                scenario = planned.case
                scenario.branch[:, column] = source.branch[:, column]
                planned_ok = True
        outages = np.sort(rng.choice(candidates, size=k, replace=False))
        result = cascade(scenario, outages, rating=rating, overload=overload)
        result.meta.update(
            {"load_scale": scale, "dispatch": dispatch, "margin": margin, "dispatched": planned_ok}
        )
        yield result
