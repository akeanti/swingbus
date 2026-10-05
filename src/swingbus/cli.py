from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from typing import Any

from swingbus import __version__
from swingbus.cascade import cascade
from swingbus.case import Case
from swingbus.contingency import n1
from swingbus.data import bundled_cases, load, pglib_cases
from swingbus.matpower import write_matpower
from swingbus.opf import rundcopf
from swingbus.powerflow import METHODS, runpf
from swingbus.report import case_info


def _emit(payload: dict[str, Any] | None, text: str, as_json: bool) -> None:
    if as_json and payload is not None:
        print(json.dumps(payload, indent=2))
    else:
        print(text)


def _prepared(name: str, from_opf: bool, rating: str) -> Case:
    case = load(name)
    if not from_opf:
        return case
    planned = rundcopf(case, rating=rating)
    if not planned.success:
        raise SystemExit(f"DC-OPF did not converge for {case.name}")
    return planned.case


def _pf(args: argparse.Namespace) -> int:
    result = runpf(
        load(args.case),
        args.method,
        tol=args.tol,
        max_iter=args.max_iter,
        enforce_q_limits=args.qlim,
        init=args.init,
    )
    if args.save:
        write_matpower(result.case, args.save)
    _emit(result.to_dict(), result.report() if args.full else result.summary(), args.json)
    return 0 if result.converged else 1


def _opf(args: argparse.Namespace) -> int:
    result = rundcopf(load(args.case), rating=args.rating)
    if args.save:
        write_matpower(result.case, args.save)
    _emit(result.to_dict(), result.summary(), args.json)
    return 0 if result.success else 1


def _n1(args: argparse.Namespace) -> int:
    case = _prepared(args.case, args.from_opf, args.rating)
    result = n1(case, method="ac" if args.ac else "dc", rating=args.rating, threshold=args.threshold)
    _emit(result.to_dict(), result.table(args.top), args.json)
    return 0


def _cascade(args: argparse.Namespace) -> int:
    case = _prepared(args.case, args.from_opf, args.rating)
    outages = [case.branch_index(f, t) for f, t in args.outage]
    result = cascade(case, outages, rating=args.rating, overload=args.overload, trip=args.trip)
    _emit(result.to_dict(), result.summary(), args.json)
    return 0


def _cases(args: argparse.Namespace) -> int:
    names = pglib_cases() if args.pglib else bundled_cases()
    print("\n".join(names))
    return 0


def _info(args: argparse.Namespace) -> int:
    print(case_info(load(args.case)))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="swingbus",
        description="Power flow, contingency analysis and DC optimal power flow on MATPOWER cases.",
    )
    parser.add_argument("--version", action="version", version=f"swingbus {__version__}")
    commands = parser.add_subparsers(dest="command", required=True, metavar="command")

    pf = commands.add_parser("pf", help="run an AC or DC power flow")
    pf.add_argument("case", help="bundled case name, PGLib-OPF name or path to a .m file")
    pf.add_argument("-m", "--method", choices=list(METHODS), default="nr")
    pf.add_argument("--tol", type=float, default=1e-8)
    pf.add_argument("--max-iter", type=int, default=None)
    pf.add_argument("--qlim", action="store_true", help="enforce generator reactive limits")
    pf.add_argument("--init", choices=["case", "flat", "dc"], default="case")
    pf.add_argument("--full", action="store_true", help="print bus and branch tables")
    pf.add_argument("--save", metavar="PATH", help="write the solved case as a MATPOWER file")
    pf.add_argument("--json", action="store_true")
    pf.set_defaults(handler=_pf)

    opf = commands.add_parser("opf", help="run a DC optimal power flow and report LMPs")
    opf.add_argument("case")
    opf.add_argument("--rating", choices=["A", "B", "C"], default="A")
    opf.add_argument("--save", metavar="PATH")
    opf.add_argument("--json", action="store_true")
    opf.set_defaults(handler=_opf)

    screening = commands.add_parser("n1", help="screen all single branch outages")
    screening.add_argument("case")
    screening.add_argument("--ac", action="store_true", help="re-solve each outage with Newton-Raphson")
    screening.add_argument("--from-opf", action="store_true", help="screen the DC-OPF dispatch")
    screening.add_argument("--rating", choices=["A", "B", "C"], default="A")
    screening.add_argument("--threshold", type=float, default=100.0, metavar="PERCENT")
    screening.add_argument("--top", type=int, default=10)
    screening.add_argument("--json", action="store_true")
    screening.set_defaults(handler=_n1)

    chain = commands.add_parser("cascade", help="simulate a DC overload cascade")
    chain.add_argument("case")
    chain.add_argument(
        "-o", "--outage", nargs=2, type=int, action="append", required=True, metavar=("FROM", "TO")
    )
    chain.add_argument("--from-opf", action="store_true")
    chain.add_argument("--rating", choices=["A", "B", "C"], default="A")
    chain.add_argument("--overload", type=float, default=1.0)
    chain.add_argument("--trip", choices=["all", "worst"], default="all")
    chain.add_argument("--json", action="store_true")
    chain.set_defaults(handler=_cascade)

    listing = commands.add_parser("cases", help="list bundled or downloadable cases")
    listing.add_argument("--pglib", action="store_true", help="list PGLib-OPF cases")
    listing.set_defaults(handler=_cases)

    info = commands.add_parser("info", help="describe a case")
    info.add_argument("case")
    info.set_defaults(handler=_info)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        code: int = args.handler(args)
    except (KeyError, ValueError, FileNotFoundError, NotImplementedError) as error:
        message = error.args[0] if isinstance(error, KeyError) and error.args else str(error)
        print(f"swingbus: error: {message}", file=sys.stderr)
        return 2
    return code
