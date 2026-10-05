from __future__ import annotations

import os
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import shortest_path

from swingbus import idx
from swingbus.case import Case, FloatArray
from swingbus.data import CaseLike, as_case

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from swingbus.opf import OPFResult
    from swingbus.powerflow import PowerFlowResult

MAX_LAYOUT_BUSES = 3000
BUS_COLOURS = {
    "vm": ("voltage magnitude (pu)", idx.VM),
    "va": ("voltage angle (deg)", idx.VA),
    "lmp": ("LMP ($/MWh)", idx.LAM_P),
}


def _pyplot() -> Any:
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ImportError("plotting needs matplotlib: pip install 'swingbus[plot]'") from error
    return plt


def _graph_distances(case: Case) -> FloatArray:
    on = case.branch[:, idx.BR_STATUS] > 0
    f = case.bus_index(case.branch[on, idx.F_BUS])
    t = case.bus_index(case.branch[on, idx.T_BUS])
    graph = sp.csr_array((np.ones(f.size), (f, t)), shape=(case.nb, case.nb))
    distances = np.asarray(shortest_path(graph, directed=False, unweighted=True), dtype=np.float64)
    finite = np.isfinite(distances)
    distances[~finite] = np.max(distances[finite]) + 1.0
    return distances


def layout(case: CaseLike, *, iterations: int = 300, seed: int = 0) -> FloatArray:
    source = as_case(case)
    n = source.nb
    if n > MAX_LAYOUT_BUSES:
        raise ValueError(f"layout is limited to {MAX_LAYOUT_BUSES} buses; pass positions explicitly")
    if n < 3:
        return np.column_stack([np.arange(n, dtype=np.float64), np.zeros(n)])
    D = _graph_distances(source)
    centering = np.eye(n) - 1.0 / n
    gram = -0.5 * centering @ (D**2) @ centering
    values, vectors = np.linalg.eigh(gram)
    X = vectors[:, -2:] * np.sqrt(np.maximum(values[-2:], 1e-9))
    X += np.random.default_rng(seed).normal(scale=1e-3, size=X.shape)
    with np.errstate(divide="ignore"):
        W = np.where(D > 0, 1.0 / D**2, 0.0)
    inverse = np.linalg.pinv(np.diag(W.sum(axis=1)) - W)
    for _ in range(iterations):
        gaps = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=-1))
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(gaps > 0, D / gaps, 0.0)
        B = -W * ratio
        B[np.diag_indices(n)] = -B.sum(axis=1)
        X = inverse @ (B @ X)
    X -= X.min(axis=0)
    return np.asarray(X / max(float(np.max(X)), 1e-12), dtype=np.float64)


def _branch_segments(case: Case, positions: FloatArray) -> tuple[list[FloatArray], Any]:
    on = np.flatnonzero(case.branch[:, idx.BR_STATUS] > 0)
    f = case.bus_index(case.branch[on, idx.F_BUS])
    t = case.bus_index(case.branch[on, idx.T_BUS])
    return list(np.stack([positions[f], positions[t]], axis=1)), on


def _bus_mode(result: object, requested: str) -> str:
    from swingbus.opf import OPFResult

    if requested != "auto":
        return requested
    if isinstance(result, OPFResult):
        return "lmp"
    method = getattr(result, "method", None)
    if method is None:
        return "none"
    return "va" if method == "DC" else "vm"


def plot_network(
    result: PowerFlowResult | OPFResult | CaseLike,
    *,
    ax: Axes | None = None,
    positions: FloatArray | None = None,
    bus_values: str = "auto",
    title: str | None = None,
) -> Axes:
    plt = _pyplot()
    from matplotlib.collections import LineCollection

    if isinstance(result, (Case, str, os.PathLike)):
        case, solved = as_case(result), False
    else:
        case, solved = result.case, True
    positions = layout(case) if positions is None else np.asarray(positions, dtype=np.float64)
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))
    segments, on = _branch_segments(case, positions)
    if solved and case.branch.shape[1] > idx.QT:
        branch = case.branch
        flow = np.maximum(np.hypot(branch[:, idx.PF], branch[:, idx.QF]), np.abs(branch[:, idx.PT]))
        rate = branch[:, idx.RATE_A]
        with np.errstate(divide="ignore", invalid="ignore"):
            loading = np.where(rate > 0, 100.0 * flow / rate, np.nan)[on]
        top = float(np.nanmax(loading)) if np.any(np.isfinite(loading)) else 100.0
        width = 0.7 + 3.5 * flow[on] / max(float(np.max(flow[on])), 1e-9)
        lines = LineCollection(segments, cmap="YlOrRd", linewidths=width.tolist(), zorder=1)
        lines.set_array(np.nan_to_num(loading, nan=0.0))
        lines.set_clim(0.0, max(100.0, top))
        ax.add_collection(lines)
        plt.colorbar(
            lines,
            ax=ax,
            orientation="horizontal",
            fraction=0.04,
            pad=0.02,
            shrink=0.7,
            label="branch loading (% of rate A)",
        )
    else:
        ax.add_collection(LineCollection(segments, colors="0.6", linewidths=0.8, zorder=1))
    has_gen = np.zeros(case.nb, dtype=bool)
    gen_on = case.gen[:, idx.GEN_STATUS] > 0
    if gen_on.any():
        has_gen[case.bus_index(case.gen[gen_on, idx.GEN_BUS])] = True
    style: dict[str, Any] = {
        "s": np.where(has_gen, 70.0, 28.0),
        "edgecolors": "black",
        "linewidths": np.where(has_gen, 1.4, 0.4).tolist(),
        "zorder": 2,
    }
    mode = _bus_mode(result, bus_values)
    if mode in BUS_COLOURS:
        label, column = BUS_COLOURS[mode]
        points = ax.scatter(positions[:, 0], positions[:, 1], c=case.bus[:, column], cmap="viridis", **style)
        plt.colorbar(points, ax=ax, fraction=0.04, pad=0.01, shrink=0.7, label=label)
    else:
        ax.scatter(positions[:, 0], positions[:, 1], color="tab:blue", **style)
    ax.set_title(title or case.name)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.autoscale_view()
    return ax


def plot_voltage(result: PowerFlowResult, *, ax: Axes | None = None) -> Axes:
    plt = _pyplot()
    case = result.case
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 3.5))
    order = np.arange(case.nb)
    low, high = case.bus[:, idx.VMIN], case.bus[:, idx.VMAX]
    ax.fill_between(order, low, high, color="tab:green", alpha=0.12, step="mid", label="limits")
    ax.plot(order, result.vm, marker="o", markersize=3, linewidth=1, color="tab:blue", label="Vm")
    ax.set_xlabel("bus (row order)")
    ax.set_ylabel("voltage magnitude (pu)")
    ax.set_title(f"{case.name} voltage profile ({result.method})")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    return ax


def plot_convergence(results: Sequence[PowerFlowResult], *, ax: Axes | None = None) -> Axes:
    plt = _pyplot()
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    for result in results:
        history = np.maximum(np.asarray(result.history), np.finfo(float).tiny)
        ax.semilogy(np.arange(history.size), history, marker="o", markersize=3, label=result.method)
    ax.set_xlabel("iteration")
    ax.set_ylabel("max power mismatch (pu)")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    return ax
