from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from swingbus import idx
from swingbus.case import Case, FloatArray, column, pad_columns
from swingbus.data import CaseLike, as_case
from swingbus.network import (
    ComplexArray,
    IntArray,
    Sparse,
    bus_types,
    check_islands,
    endpoints,
    gen_buses,
    make_bdc,
    make_sbus,
    make_ybus,
    to_external,
    to_internal,
)
from swingbus.solvers import Solution, dc_angles, fast_decoupled, gauss_seidel, newton

METHODS = {
    "nr": "Newton-Raphson",
    "fdxb": "fast-decoupled XB",
    "fdbx": "fast-decoupled BX",
    "gs": "Gauss-Seidel",
    "dc": "DC",
}
MAX_ITER = {"nr": 10, "fdxb": 30, "fdbx": 30, "gs": 1000}
INITS = ("case", "flat", "dc")
Q_TOLERANCE = 1e-6


@dataclass(eq=False)
class PowerFlowResult:
    case: Case
    method: str
    converged: bool
    iterations: int
    history: list[float]
    elapsed: float
    q_limited: list[int] = field(default_factory=list)

    @property
    def vm(self) -> FloatArray:
        return column(self.case.bus, idx.VM)

    @property
    def va(self) -> FloatArray:
        return column(self.case.bus, idx.VA)

    @property
    def pg(self) -> FloatArray:
        return column(self.case.gen, idx.PG)

    @property
    def qg(self) -> FloatArray:
        return column(self.case.gen, idx.QG)

    @property
    def pf(self) -> FloatArray:
        return column(self.case.branch, idx.PF)

    @property
    def qf(self) -> FloatArray:
        return column(self.case.branch, idx.QF)

    @property
    def pt(self) -> FloatArray:
        return column(self.case.branch, idx.PT)

    @property
    def qt(self) -> FloatArray:
        return column(self.case.branch, idx.QT)

    @property
    def flow(self) -> FloatArray:
        return np.maximum(np.hypot(self.pf, self.qf), np.hypot(self.pt, self.qt))

    def loading(self, rating: str = "A") -> FloatArray:
        rate = column(self.case.branch, idx.RATINGS[rating.upper()])
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(rate > 0, 100.0 * self.flow / rate, np.nan)

    @property
    def losses(self) -> complex:
        branch = self.case.branch
        on = branch[:, idx.BR_STATUS] > 0
        return complex(
            np.sum(branch[on, idx.PF] + branch[on, idx.PT])
            + 1j * np.sum(branch[on, idx.QF] + branch[on, idx.QT])
        )

    @property
    def generation(self) -> complex:
        on = self.case.gen[:, idx.GEN_STATUS] > 0
        return complex(np.sum(self.pg[on]) + 1j * np.sum(self.qg[on]))

    @property
    def load(self) -> complex:
        energized = self.case.bus[:, idx.BUS_TYPE] != idx.NONE
        bus = self.case.bus[energized]
        return complex(np.sum(bus[:, idx.PD]) + 1j * np.sum(bus[:, idx.QD]))

    @property
    def shunts(self) -> complex:
        energized = self.case.bus[:, idx.BUS_TYPE] != idx.NONE
        bus = self.case.bus[energized]
        squared = bus[:, idx.VM] ** 2
        return complex(np.sum(bus[:, idx.GS] * squared) - 1j * np.sum(bus[:, idx.BS] * squared))

    def summary(self) -> str:
        from swingbus.report import powerflow_summary

        return powerflow_summary(self)

    def report(self) -> str:
        from swingbus.report import powerflow_report

        return powerflow_report(self)

    def to_dict(self) -> dict[str, Any]:
        from swingbus.report import powerflow_dict

        return powerflow_dict(self)

    def __str__(self) -> str:
        return self.summary()


def initial_voltage(case: Case, ref: IntArray, pv: IntArray, pq: IntArray, init: str) -> ComplexArray:
    bus = case.bus
    if init == "flat":
        Vm = np.ones(case.nb)
        Va = np.zeros(case.nb)
        Va[ref] = np.deg2rad(bus[ref, idx.VA])
    elif init == "dc":
        Vm = np.ones(case.nb)
        Va = dc_state(case, ref, pv, pq)[0]
    else:
        Vm = np.where(bus[:, idx.VM] > 0, bus[:, idx.VM], 1.0)
        Va = np.deg2rad(bus[:, idx.VA])
    V0 = Vm * np.exp(1j * Va)
    on = case.gen[:, idx.GEN_STATUS] > 0
    gbus = gen_buses(case)[on]
    controlled = np.ones(case.nb, dtype=bool)
    controlled[pq] = False
    regulated = controlled[gbus]
    target = gbus[regulated]
    V0[target] = case.gen[on, idx.VG][regulated] * V0[target] / np.abs(V0[target])
    return np.asarray(V0, dtype=np.complex128)


def _split_reactive(gen: FloatArray, on: IntArray, gbus: IntArray, nb: int) -> None:
    count = np.bincount(gbus, minlength=nb)
    share = count[gbus]
    multi = share > 1
    if not multi.any():
        return
    qg = gen[on, idx.QG] / share
    qmin = gen[on, idx.QMIN].copy()
    qmax = gen[on, idx.QMAX].copy()
    proxy = np.abs(qg) + np.where(np.isinf(qmax), 0.0, np.abs(qmax))
    proxy += np.where(np.isinf(qmin), 0.0, np.abs(qmin))
    proxy = np.bincount(gbus, weights=proxy, minlength=nb)[gbus]
    qmin = np.where(np.isinf(qmin), np.sign(qmin) * proxy, qmin)
    qmax = np.where(np.isinf(qmax), np.sign(qmax) * proxy, qmax)
    total = np.bincount(gbus, weights=qg, minlength=nb)
    low = np.bincount(gbus, weights=qmin, minlength=nb)
    high = np.bincount(gbus, weights=qmax, minlength=nb)
    split = qmin + ((total - low) / (high - low + np.finfo(float).eps))[gbus] * (qmax - qmin)
    flat = low[gbus] == high[gbus]
    split[flat] = qmin[flat] + ((total - low) / np.maximum(count, 1))[gbus][flat]
    gen[on[multi], idx.QG] = split[multi]


def apply_solution(case: Case, Ybus: Sparse, Yf: Sparse, Yt: Sparse, V: ComplexArray, ref: IntArray) -> None:
    base = case.base_mva
    bus, gen = case.bus, case.gen
    bus[:, idx.VM] = np.abs(V)
    bus[:, idx.VA] = np.rad2deg(np.angle(V))
    gbus_all = gen_buses(case)
    on = np.flatnonzero((gen[:, idx.GEN_STATUS] > 0) & (bus[gbus_all, idx.BUS_TYPE] != idx.PQ))
    gbus = gbus_all[on]
    injection = V[gbus] * np.conj(Ybus[gbus] @ V)
    gen[on, idx.QG] = injection.imag * base + bus[gbus, idx.QD]
    _split_reactive(gen, on, gbus, case.nb)
    for r in ref.tolist():
        at_ref = on[gbus == r]
        if at_ref.size:
            s = V[r] * np.conj(Ybus[[r]] @ V)[0]
            others = gen[at_ref[1:], idx.PG].sum()
            gen[at_ref[0], idx.PG] = s.real * base + bus[r, idx.PD] - others
    f, t = endpoints(case)
    sf = V[f] * np.conj(Yf @ V) * base
    st = V[t] * np.conj(Yt @ V) * base
    case.branch = pad_columns(case.branch, idx.BRANCH_COLS_PF)
    case.branch[:, idx.PF] = sf.real
    case.branch[:, idx.QF] = sf.imag
    case.branch[:, idx.PT] = st.real
    case.branch[:, idx.QT] = st.imag


def dc_state(
    case: Case, ref: IntArray, pv: IntArray, pq: IntArray
) -> tuple[FloatArray, Sparse, Sparse, FloatArray, FloatArray]:
    Bbus, Bf, Pbusinj, Pfinj = make_bdc(case)
    Pbus = make_sbus(case).real - Pbusinj - case.bus[:, idx.GS] / case.base_mva
    Va = dc_angles(Bbus, Pbus, np.deg2rad(case.bus[:, idx.VA]), ref, pv, pq)
    return Va, Bbus, Bf, Pbus, Pfinj


def apply_dc_solution(case: Case, ref: IntArray, pv: IntArray, pq: IntArray) -> None:
    base = case.base_mva
    Va, Bbus, Bf, Pbus, Pfinj = dc_state(case, ref, pv, pq)
    case.bus[:, idx.VM] = 1.0
    case.bus[:, idx.VA] = np.rad2deg(Va)
    case.branch = pad_columns(case.branch, idx.BRANCH_COLS_PF)
    flow = (Bf @ Va + Pfinj) * base
    case.branch[:, idx.PF] = flow
    case.branch[:, idx.PT] = -flow
    case.branch[:, idx.QF] = 0.0
    case.branch[:, idx.QT] = 0.0
    on = case.gen[:, idx.GEN_STATUS] > 0
    gbus = gen_buses(case)
    for r in ref.tolist():
        at_ref = np.flatnonzero(on & (gbus == r))
        if at_ref.size:
            case.gen[at_ref[0], idx.PG] += ((Bbus[[r]] @ Va)[0] - Pbus[r]) * base


def _solve(
    method: str,
    case: Case,
    Ybus: Sparse,
    Sbus: ComplexArray,
    V0: ComplexArray,
    ref: IntArray,
    pv: IntArray,
    pq: IntArray,
    tol: float,
    max_iter: int,
) -> Solution:
    if method == "nr":
        return newton(Ybus, Sbus, V0, ref, pv, pq, tol, max_iter)
    if method == "gs":
        return gauss_seidel(Ybus, Sbus, V0, ref, pv, pq, tol, max_iter)
    Bp, Bpp = make_b(case, method)
    return fast_decoupled(Ybus, Sbus, V0, ref, pv, pq, Bp, Bpp, tol, max_iter)


def make_b(case: Case, method: str) -> tuple[Sparse, Sparse]:
    prime = case.copy()
    prime.bus[:, idx.BS] = 0.0
    prime.branch[:, idx.BR_B] = 0.0
    prime.branch[:, idx.TAP] = 1.0
    if method == "fdxb":
        prime.branch[:, idx.BR_R] = 0.0
    double = case.copy()
    double.branch[:, idx.SHIFT] = 0.0
    if method == "fdbx":
        double.branch[:, idx.BR_R] = 0.0
    return -make_ybus(prime)[0].imag, -make_ybus(double)[0].imag


def _q_violations(case: Case, pv: IntArray) -> tuple[IntArray, FloatArray]:
    gen = case.gen
    candidates = (gen[:, idx.GEN_STATUS] > 0) & np.isin(gen_buses(case), pv)
    over = candidates & (gen[:, idx.QG] > gen[:, idx.QMAX] + Q_TOLERANCE)
    under = candidates & (gen[:, idx.QG] < gen[:, idx.QMIN] - Q_TOLERANCE)
    violated = np.flatnonzero(over | under)
    bound = np.where(over, gen[:, idx.QMAX], gen[:, idx.QMIN])[violated]
    return violated, bound


def runpf(
    case: CaseLike,
    method: str = "nr",
    *,
    tol: float = 1e-8,
    max_iter: int | None = None,
    enforce_q_limits: bool = False,
    init: str = "case",
) -> PowerFlowResult:
    source = as_case(case)
    method = method.lower()
    if method == "dc":
        return rundcpf(source)
    if method not in MAX_ITER:
        raise ValueError(f"unknown method {method!r}, expected one of {', '.join(METHODS)}")
    if init not in INITS:
        raise ValueError(f"unknown init {init!r}, expected one of {', '.join(INITS)}")
    limit = MAX_ITER[method] if max_iter is None else max_iter
    start = time.perf_counter()
    work, index = to_internal(source)
    original_types = work.bus[:, idx.BUS_TYPE].copy()
    fixed_q = np.zeros(work.ng)
    limited: list[int] = []
    history: list[float] = []
    iterations = 0
    start_from = init
    while True:
        ref, pv, pq = bus_types(work)
        check_islands(work, ref)
        Ybus, Yf, Yt = make_ybus(work)
        V0 = initial_voltage(work, ref, pv, pq, start_from)
        solution = _solve(method, work, Ybus, make_sbus(work), V0, ref, pv, pq, tol, limit)
        iterations += solution.iterations
        history.extend(solution.history)
        apply_solution(work, Ybus, Yf, Yt, solution.V, ref)
        if not (solution.converged and enforce_q_limits):
            break
        violated, bound = _q_violations(work, pv)
        if violated.size == 0:
            break
        for g, q in zip(violated.tolist(), bound.tolist(), strict=True):
            b = int(work.gen[g, idx.GEN_BUS])
            fixed_q[g] = q
            work.gen[g, idx.QG] = q
            work.gen[g, idx.GEN_STATUS] = 0
            work.bus[b, idx.PD] -= work.gen[g, idx.PG]
            work.bus[b, idx.QD] -= q
            work.bus[b, idx.BUS_TYPE] = idx.PQ
        limited.extend(violated.tolist())
        start_from = "case"
    for g in limited:
        b = int(work.gen[g, idx.GEN_BUS])
        work.bus[b, idx.PD] += work.gen[g, idx.PG]
        work.bus[b, idx.QD] += fixed_q[g]
        work.gen[g, idx.QG] = fixed_q[g]
        work.gen[g, idx.GEN_STATUS] = 1
    work.bus[:, idx.BUS_TYPE] = original_types
    return PowerFlowResult(
        to_external(source, work, index),
        METHODS[method],
        solution.converged,
        iterations,
        history,
        time.perf_counter() - start,
        sorted(index.gen_rows[limited].tolist()),
    )


def rundcpf(case: CaseLike) -> PowerFlowResult:
    source = as_case(case)
    start = time.perf_counter()
    work, index = to_internal(source)
    ref, pv, pq = bus_types(work)
    check_islands(work, ref)
    apply_dc_solution(work, ref, pv, pq)
    return PowerFlowResult(
        to_external(source, work, index),
        METHODS["dc"],
        True,
        1,
        [],
        time.perf_counter() - start,
    )
