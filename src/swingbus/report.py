from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from swingbus import idx
from swingbus.case import Case, FloatArray
from swingbus.network import bridges, endpoints, islands, to_internal

if TYPE_CHECKING:
    from swingbus.cascade import CascadeResult
    from swingbus.contingency import ContingencyResult
    from swingbus.opf import OPFResult
    from swingbus.powerflow import PowerFlowResult


def table(headers: Sequence[str], rows: Sequence[Sequence[str]], align: str | None = None) -> str:
    align = align or "r" * len(headers)
    widths = [len(h) for h in headers]
    for row in rows:
        widths = [max(width, len(cell)) for width, cell in zip(widths, row, strict=True)]

    def line(cells: Sequence[str]) -> str:
        parts = [
            cell.ljust(width) if side == "l" else cell.rjust(width)
            for cell, width, side in zip(cells, widths, align, strict=True)
        ]
        return "  ".join(parts).rstrip()

    rule = "  ".join("-" * width for width in widths)
    return "\n".join([line(headers), rule, *(line(row) for row in rows)])


def _f(value: float, digits: int = 3) -> str:
    if not np.isfinite(value):
        return "-"
    return f"{round(float(value), digits) + 0.0:.{digits}f}"


def _count(number: int, singular: str, plural: str | None = None) -> str:
    word = singular if number == 1 else (plural or singular + "s")
    return f"{number} {word}"


def _bus(case: Case, row: int) -> int:
    return int(case.bus[row, idx.BUS_I])


def _branch_label(case: Case, row: int) -> str:
    return f"{int(case.branch[row, idx.F_BUS])}-{int(case.branch[row, idx.T_BUS])}"


def _ms(seconds: float) -> str:
    return f"{seconds * 1e3:.1f} ms"


def _energized(case: Case) -> NDArray[np.bool_]:
    return np.asarray(case.bus[:, idx.BUS_TYPE] != idx.NONE, dtype=bool)


def powerflow_summary(result: PowerFlowResult) -> str:
    case = result.case
    status = (
        f"converged in {result.iterations} iterations"
        if result.converged
        else f"did not converge after {result.iterations} iterations"
    )
    if result.method == "DC":
        status = "solved"
    head = f"{case.name} | {result.method} | {status} ({_ms(result.elapsed)})"
    size = (
        f"{_count(case.nb, 'bus', 'buses')}, {_count(case.ng, 'generator')}, "
        f"{_count(case.nl, 'branch', 'branches')}, base {case.base_mva:g} MVA"
    )
    gen, load, loss, shunt = result.generation, result.load, result.losses, result.shunts
    balance_rows = [
        ["generation", _f(gen.real), _f(gen.imag)],
        ["load", _f(load.real), _f(load.imag)],
        ["branch losses", _f(loss.real), _f(loss.imag)],
    ]
    if abs(shunt) > 0:
        balance_rows.append(["bus shunts", _f(shunt.real), _f(shunt.imag)])
    balance = table(["", "P (MW)", "Q (MVAr)"], balance_rows, "lrr")
    energized = np.flatnonzero(_energized(case))
    vm, va = result.vm[energized], result.va[energized]
    low, high = energized[int(np.argmin(vm))], energized[int(np.argmax(vm))]
    far = energized[int(np.argmax(np.abs(va)))]
    lines = [
        head,
        size,
        "",
        balance,
        "",
        f"voltage  min {np.min(vm):.4f} pu at bus {_bus(case, low)}, "
        f"max {np.max(vm):.4f} pu at bus {_bus(case, high)}",
        f"angle    largest {result.va[far]:.3f} deg at bus {_bus(case, far)}",
    ]
    loading = result.loading()
    if np.any(np.isfinite(loading)):
        worst = int(np.nanargmax(loading))
        over = int(np.sum(loading > 100.0))
        lines.append(
            f"loading  max {loading[worst]:.1f} % on branch {_branch_label(case, worst)}, {over} above rate A"
        )
    if result.q_limited:
        listed = ", ".join(str(int(case.gen[g, idx.GEN_BUS])) for g in result.q_limited)
        lines.append(f"q limits generators held at a limit on buses {listed}")
    return "\n".join(lines)


def powerflow_report(result: PowerFlowResult) -> str:
    case = result.case
    on = case.gen[:, idx.GEN_STATUS] > 0
    gen_p = np.zeros(case.nb)
    gen_q = np.zeros(case.nb)
    rows = case.bus_index(case.gen[on, idx.GEN_BUS]) if on.any() else np.zeros(0, dtype=np.intp)
    np.add.at(gen_p, rows, case.gen[on, idx.PG])
    np.add.at(gen_q, rows, case.gen[on, idx.QG])
    has_gen = np.zeros(case.nb, dtype=bool)
    has_gen[rows] = True
    bus_rows = [
        [
            str(_bus(case, row)),
            _f(case.bus[row, idx.VM], 4),
            _f(case.bus[row, idx.VA]),
            _f(gen_p[row], 2) if has_gen[row] else "",
            _f(gen_q[row], 2) if has_gen[row] else "",
            _f(case.bus[row, idx.PD], 2),
            _f(case.bus[row, idx.QD], 2),
        ]
        for row in range(case.nb)
    ]
    loading = result.loading()
    branch_rows = [
        [
            str(row + 1),
            str(int(b[idx.F_BUS])),
            str(int(b[idx.T_BUS])),
            _f(b[idx.PF], 2),
            _f(b[idx.QF], 2),
            _f(b[idx.PT], 2),
            _f(b[idx.QT], 2),
            _f(b[idx.PF] + b[idx.PT], 3),
            _f(loading[row], 1),
        ]
        for row, b in enumerate(case.branch)
    ]
    return "\n\n".join(
        [
            powerflow_summary(result),
            table(["bus", "Vm (pu)", "Va (deg)", "Pg (MW)", "Qg (MVAr)", "Pd (MW)", "Qd (MVAr)"], bus_rows),
            table(
                ["#", "from", "to", "Pf (MW)", "Qf (MVAr)", "Pt (MW)", "Qt (MVAr)", "loss (MW)", "load (%)"],
                branch_rows,
            ),
        ]
    )


def _clean(values: FloatArray) -> list[float | None]:
    return [float(v) if np.isfinite(v) else None for v in values]


def powerflow_dict(result: PowerFlowResult) -> dict[str, Any]:
    case = result.case
    gen, load, loss, shunt = result.generation, result.load, result.losses, result.shunts
    return {
        "case": case.name,
        "method": result.method,
        "converged": result.converged,
        "iterations": result.iterations,
        "elapsed_ms": result.elapsed * 1e3,
        "base_mva": case.base_mva,
        "totals": {
            "generation_mw": gen.real,
            "generation_mvar": gen.imag,
            "load_mw": load.real,
            "load_mvar": load.imag,
            "losses_mw": loss.real,
            "losses_mvar": loss.imag,
            "shunts_mw": shunt.real,
            "shunts_mvar": shunt.imag,
        },
        "bus": {
            "id": case.bus[:, idx.BUS_I].astype(int).tolist(),
            "vm": _clean(result.vm),
            "va": _clean(result.va),
            "pd": _clean(case.bus[:, idx.PD]),
            "qd": _clean(case.bus[:, idx.QD]),
        },
        "gen": {
            "bus": case.gen[:, idx.GEN_BUS].astype(int).tolist(),
            "status": case.gen[:, idx.GEN_STATUS].astype(int).tolist(),
            "pg": _clean(result.pg),
            "qg": _clean(result.qg),
        },
        "branch": {
            "from": case.branch[:, idx.F_BUS].astype(int).tolist(),
            "to": case.branch[:, idx.T_BUS].astype(int).tolist(),
            "status": case.branch[:, idx.BR_STATUS].astype(int).tolist(),
            "pf": _clean(result.pf),
            "qf": _clean(result.qf),
            "pt": _clean(result.pt),
            "qt": _clean(result.qt),
            "loading": _clean(result.loading()),
        },
        "q_limited": result.q_limited,
        "history": result.history,
    }


def _status(result: ContingencyResult, position: int) -> str:
    if result.islanding[position]:
        return "islanding"
    if not result.converged[position]:
        return "diverged"
    if result.overloads[position] or result.voltage_violations[position]:
        return "violation"
    return "ok"


def contingency_table(result: ContingencyResult, top: int = 10) -> str:
    case = result.case
    n = result.outages.size
    head = (
        f"N-1 {result.method} screening of {case.name} | {n} outages | "
        f"{int(result.islanding.sum())} islanding | "
        f"{int(np.sum(result.overloads > 0))} with overloads | "
        f"{int(np.sum(~result.converged))} diverged | {_ms(result.elapsed)}"
    )
    ac = result.method == "AC"
    headers = ["rank", "outage", "max load (%)", "worst branch", "overloads"]
    if ac:
        headers += ["Vmin (pu)", "Vmax (pu)", "V viol."]
    headers.append("status")
    rows = []
    for rank, position in enumerate(result.ranking()[:top].tolist(), start=1):
        worst = int(result.worst_branch[position])
        row = [
            str(rank),
            _branch_label(case, int(result.outages[position])),
            _f(result.max_loading[position], 1),
            _branch_label(case, worst) if worst >= 0 else "-",
            str(int(result.overloads[position])),
        ]
        if ac:
            row += [
                _f(result.vmin[position], 4),
                _f(result.vmax[position], 4),
                str(int(result.voltage_violations[position])),
            ]
        row.append(_status(result, position))
        rows.append(row)
    align = "rlrlr" + ("rrr" if ac else "") + "l"
    return head + "\n\n" + table(headers, rows, align)


def contingency_dict(result: ContingencyResult) -> dict[str, Any]:
    case = result.case
    return {
        "case": case.name,
        "method": result.method,
        "rating": result.rating,
        "threshold": result.threshold,
        "elapsed_ms": result.elapsed * 1e3,
        "outages": [
            {
                "branch": int(k),
                "from": int(case.branch[k, idx.F_BUS]),
                "to": int(case.branch[k, idx.T_BUS]),
                "status": _status(result, position),
                "max_loading": _clean(result.max_loading[[position]])[0],
                "worst_branch": int(result.worst_branch[position]),
                "overloads": int(result.overloads[position]),
                "vmin": _clean(result.vmin[[position]])[0],
                "vmax": _clean(result.vmax[[position]])[0],
                "voltage_violations": int(result.voltage_violations[position]),
            }
            for position, k in enumerate(result.outages.tolist())
        ],
    }


def opf_summary(result: OPFResult) -> str:
    case = result.case
    state = "optimal" if result.success else "not converged"
    head = (
        f"{case.name} | DC-OPF | {state} | cost {result.objective:.2f} $/h | "
        f"{result.iterations} iterations ({_ms(result.elapsed)})"
    )
    on = np.flatnonzero(case.gen[:, idx.GEN_STATUS] > 0)
    gen_rows = [
        [
            str(g + 1),
            str(int(case.gen[g, idx.GEN_BUS])),
            _f(case.gen[g, idx.PG], 2),
            _f(case.gen[g, idx.PMIN], 2),
            _f(case.gen[g, idx.PMAX], 2),
            _f(case.gen[g, idx.MU_PMAX] - case.gen[g, idx.MU_PMIN], 4),
        ]
        for g in on.tolist()
    ]
    energized = np.flatnonzero(_energized(case))
    lmp = result.lmp[energized]
    cheapest = _bus(case, int(energized[int(np.argmin(lmp))]))
    dearest = _bus(case, int(energized[int(np.argmax(lmp))]))
    lines = [
        head,
        "",
        table(["gen", "bus", "Pg (MW)", "Pmin (MW)", "Pmax (MW)", "limit price ($/MWh)"], gen_rows),
        "",
        f"LMP  min {np.min(lmp):.4f} $/MWh at bus {cheapest}, max {np.max(lmp):.4f} $/MWh at bus {dearest}",
    ]
    price = result.congestion_price
    congested = np.flatnonzero(price > 1e-6)
    if congested.size:
        rows = [
            [
                _branch_label(case, k),
                _f(result.pf[k], 2),
                _f(case.branch[k, idx.RATE_A], 2),
                _f(price[k], 4),
            ]
            for k in congested[np.argsort(-price[congested])].tolist()
        ]
        headers = ["congested branch", "flow (MW)", "limit (MW)", "shadow price ($/MWh)"]
        lines += ["", table(headers, rows, "lrrr")]
    else:
        lines.append("no branch limits are binding")
    return "\n".join(lines)


def opf_dict(result: OPFResult) -> dict[str, Any]:
    case = result.case
    return {
        "case": case.name,
        "success": result.success,
        "objective": result.objective,
        "iterations": result.iterations,
        "elapsed_ms": result.elapsed * 1e3,
        "bus": {
            "id": case.bus[:, idx.BUS_I].astype(int).tolist(),
            "lmp": _clean(result.lmp),
            "va": _clean(result.va),
        },
        "gen": {"bus": case.gen[:, idx.GEN_BUS].astype(int).tolist(), "pg": _clean(result.pg)},
        "branch": {
            "from": case.branch[:, idx.F_BUS].astype(int).tolist(),
            "to": case.branch[:, idx.T_BUS].astype(int).tolist(),
            "pf": _clean(result.pf),
            "congestion_price": _clean(result.congestion_price),
        },
    }


def cascade_summary(result: CascadeResult) -> str:
    case = result.case
    initial = ", ".join(_branch_label(case, int(k)) for k in result.initial)
    lines = [
        f"{case.name} | DC cascade | initial outage {initial}",
        f"stages {len(result.stages)}, branches tripped {result.tripped.size}, islands {result.islands}",
        f"demand {result.demand:.2f} MW, served {result.served:.2f} MW, "
        f"not served {result.shed:.2f} MW ({100.0 * result.shed / max(result.demand, 1e-12):.1f} %)",
    ]
    for number, stage in enumerate(result.stages, start=1):
        lines.append(f"  stage {number}: " + ", ".join(_branch_label(case, int(k)) for k in stage))
    return "\n".join(lines)


def cascade_dict(result: CascadeResult) -> dict[str, Any]:
    return {
        "case": result.case.name,
        "initial": result.initial.tolist(),
        "stages": [stage.tolist() for stage in result.stages],
        "demand_mw": result.demand,
        "served_mw": result.served,
        "not_served_mw": result.shed,
        "islands": result.islands,
        "meta": result.meta,
    }


def case_info(case: Case) -> str:
    work, _ = to_internal(case)
    f, t = endpoints(work)
    branch = case.branch
    transformers = int(np.sum((branch[:, idx.TAP] != 0) | (branch[:, idx.SHIFT] != 0)))
    on = case.gen[:, idx.GEN_STATUS] > 0
    energized = _energized(case)
    levels = sorted({float(kv) for kv in case.bus[:, idx.BASE_KV] if kv > 0})
    in_service = int(np.sum(branch[:, idx.BR_STATUS] > 0))
    rows = [
        ["buses", str(case.nb)],
        ["generators", f"{case.ng} ({int(on.sum())} in service)"],
        ["branches", f"{case.nl} ({in_service} in service, {transformers} transformers)"],
        ["islands", str(int(np.unique(islands(work)).size))],
        ["radial branches", str(int(bridges(work.nb, f, t).sum()))],
        ["load", f"{case.bus[energized, idx.PD].sum():.2f} MW, {case.bus[energized, idx.QD].sum():.2f} MVAr"],
        ["capacity", f"{case.gen[on, idx.PMAX].sum():.2f} MW"],
        ["rated branches", str(int(np.sum(branch[:, idx.RATE_A] > 0)))],
        ["voltage levels", ", ".join(f"{kv:g} kV" for kv in levels) or "not given"],
        ["costs", "yes" if case.gencost is not None else "no"],
    ]
    return f"{case.name} (base {case.base_mva:g} MVA)\n\n" + table(["", ""], rows, "ll").split("\n", 2)[2]
