from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from swingbus import idx
from swingbus.case import Case, FloatArray
from swingbus.data import CaseLike, as_case
from swingbus.network import IntArray, bridges, endpoints, to_internal
from swingbus.powerflow import rundcpf, runpf
from swingbus.sensitivity import lodf, ptdf

BoolArray = NDArray[np.bool_]
CHUNK = 256
VOLTAGE_TOLERANCE = 1e-4


@dataclass(eq=False)
class ContingencyResult:
    case: Case
    method: str
    rating: str
    threshold: float
    outages: IntArray
    islanding: BoolArray
    converged: BoolArray
    max_loading: FloatArray
    worst_branch: IntArray
    overloads: IntArray
    vmin: FloatArray
    vmax: FloatArray
    voltage_violations: IntArray
    base_loading: FloatArray
    elapsed: float

    @property
    def insecure(self) -> BoolArray:
        return self.islanding | ~self.converged | (self.overloads > 0) | (self.voltage_violations > 0)

    def ranking(self) -> IntArray:
        diverged = ~self.converged & ~self.islanding
        severity = np.where(diverged, np.inf, np.nan_to_num(self.max_loading, nan=-1.0))
        severity = np.where(self.islanding, -2.0, severity)
        return np.argsort(-severity, kind="stable")

    def table(self, top: int = 10) -> str:
        from swingbus.report import contingency_table

        return contingency_table(self, top)

    def to_dict(self) -> dict[str, Any]:
        from swingbus.report import contingency_dict

        return contingency_dict(self)

    def __str__(self) -> str:
        return self.table()


def _selection(source: Case, branches: ArrayLike | None) -> IntArray:
    in_service = to_internal(source)[1].branch_rows
    if branches is None:
        return in_service
    chosen = np.atleast_1d(np.asarray(branches, dtype=np.intp))
    return np.intersect1d(chosen, in_service)


def _bridge_mask(source: Case) -> BoolArray:
    work, index = to_internal(source)
    f, t = endpoints(work)
    mask = np.zeros(source.nl, dtype=bool)
    mask[index.branch_rows] = bridges(work.nb, f, t)
    return mask


def _empty(n: int, value: float) -> FloatArray:
    return np.full(n, value, dtype=np.float64)


def _n1_dc(source: Case, rate: FloatArray, threshold: float, outages: IntArray) -> dict[str, Any]:
    base = rundcpf(source)
    flow = base.pf
    L = lodf(source, ptdf(source))
    limited = np.flatnonzero(rate > 0)
    n = outages.size
    max_loading = _empty(n, np.nan)
    worst = np.full(n, -1, dtype=np.intp)
    overloads = np.zeros(n, dtype=np.intp)
    islanding = np.isnan(L[:, outages]).any(axis=0) if n else np.zeros(0, dtype=bool)
    for start in range(0, n, CHUNK):
        cols = outages[start : start + CHUNK]
        post = flow[:, None] + L[:, cols] * flow[cols][None, :]
        loading = 100.0 * np.abs(post[limited]) / rate[limited][:, None]
        span = slice(start, start + cols.size)
        if limited.size:
            best = np.nanargmax(np.nan_to_num(loading, nan=-np.inf), axis=0)
            max_loading[span] = loading[best, np.arange(cols.size)]
            worst[span] = limited[best]
            overloads[span] = np.sum(loading > threshold, axis=0)
    max_loading[islanding] = np.nan
    worst[islanding] = -1
    overloads[islanding] = 0
    base_loading = np.where(rate > 0, 100.0 * np.abs(flow) / np.where(rate > 0, rate, 1.0), np.nan)
    return {
        "islanding": islanding,
        "converged": np.ones(n, dtype=bool),
        "max_loading": max_loading,
        "worst_branch": worst,
        "overloads": overloads,
        "vmin": _empty(n, np.nan),
        "vmax": _empty(n, np.nan),
        "voltage_violations": np.zeros(n, dtype=np.intp),
        "base_loading": base_loading,
    }


def _n1_ac(
    source: Case, label: str, threshold: float, outages: IntArray, tol: float, max_iter: int
) -> dict[str, Any]:
    base = runpf(source, "nr", tol=tol, max_iter=max_iter)
    if not base.converged:
        raise RuntimeError(f"base case power flow of {source.name} did not converge")
    warm = base.case
    energized = warm.bus[:, idx.BUS_TYPE] != idx.NONE
    vmin_limit = warm.bus[:, idx.VMIN]
    vmax_limit = warm.bus[:, idx.VMAX]
    banded = energized & (vmax_limit > 0)
    is_bridge = _bridge_mask(source)
    n = outages.size
    out = {
        "islanding": is_bridge[outages],
        "converged": np.zeros(n, dtype=bool),
        "max_loading": _empty(n, np.nan),
        "worst_branch": np.full(n, -1, dtype=np.intp),
        "overloads": np.zeros(n, dtype=np.intp),
        "vmin": _empty(n, np.nan),
        "vmax": _empty(n, np.nan),
        "voltage_violations": np.zeros(n, dtype=np.intp),
        "base_loading": base.loading(label),
    }
    for position, k in enumerate(outages.tolist()):
        if out["islanding"][position]:
            continue
        trial = warm.copy()
        trial.branch[k, idx.BR_STATUS] = 0
        result = runpf(trial, "nr", tol=tol, max_iter=max_iter)
        if not result.converged:
            continue
        out["converged"][position] = True
        loading = result.loading(label)
        loading[k] = np.nan
        if np.any(np.isfinite(loading)):
            worst = int(np.nanargmax(loading))
            out["max_loading"][position] = loading[worst]
            out["worst_branch"][position] = worst
            out["overloads"][position] = int(np.sum(loading > threshold))
        vm = result.vm
        out["vmin"][position] = float(np.min(vm[energized]))
        out["vmax"][position] = float(np.max(vm[energized]))
        low = vm < vmin_limit - VOLTAGE_TOLERANCE
        high = vm > vmax_limit + VOLTAGE_TOLERANCE
        out["voltage_violations"][position] = int(np.sum(banded & (low | high)))
    return out


def n1(
    case: CaseLike,
    *,
    method: str = "dc",
    rating: str = "A",
    threshold: float = 100.0,
    branches: ArrayLike | None = None,
    tol: float = 1e-8,
    max_iter: int = 20,
) -> ContingencyResult:
    source = as_case(case)
    method = method.lower()
    if method not in {"dc", "ac"}:
        raise ValueError(f"unknown method {method!r}, expected 'dc' or 'ac'")
    label = rating.upper()
    if label not in idx.RATINGS:
        raise ValueError(f"unknown rating {rating!r}, expected A, B or C")
    rate = source.branch[:, idx.RATINGS[label]].copy()
    outages = _selection(source, branches)
    start = time.perf_counter()
    if method == "dc":
        fields = _n1_dc(source, rate, threshold, outages)
    else:
        fields = _n1_ac(source, label, threshold, outages, tol, max_iter)
    return ContingencyResult(
        case=source,
        method=method.upper(),
        rating=rating.upper(),
        threshold=threshold,
        outages=outages,
        elapsed=time.perf_counter() - start,
        **fields,
    )
