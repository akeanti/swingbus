from __future__ import annotations

import json
import subprocess
import sys

import pytest

import swingbus as sb
from swingbus.cli import main


def run(capsys, *args):
    code = main(list(args))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_power_flow_summary(capsys):
    code, out, _ = run(capsys, "pf", "case14")
    assert code == 0
    assert out.startswith("case14 | Newton-Raphson | converged in 2 iterations")
    assert "generation" in out


def test_power_flow_json_and_save(capsys, tmp_path):
    target = tmp_path / "solved.m"
    code, out, _ = run(capsys, "pf", "case118", "-m", "fdxb", "--qlim", "--json", "--save", str(target))
    assert code == 0
    payload = json.loads(out)
    assert payload["method"] == "fast-decoupled XB"
    assert payload["q_limited"]
    assert sb.load(target).nb == 118


def test_full_report_and_nonconvergence(capsys):
    code, out, _ = run(capsys, "pf", "case14", "--full")
    assert "Qd (MVAr)" in out
    code, out, _ = run(capsys, "pf", "case118", "--max-iter", "1", "--init", "flat")
    assert code == 1


def test_opf(capsys):
    code, out, _ = run(capsys, "opf", "pglib_opf_case5_pjm")
    assert code == 0
    assert "17479.90 $/h" in out


def test_n1_and_cascade(capsys):
    code, out, _ = run(capsys, "n1", "pglib_opf_case24_ieee_rts", "--from-opf", "--top", "3")
    assert code == 0
    assert "islanding" in out
    code, out, _ = run(capsys, "n1", "case14", "--ac", "--json")
    assert json.loads(out)["method"] == "AC"
    outages = ["-o", "8", "5", "-o", "26", "30"]
    code, out, _ = run(capsys, "cascade", "pglib_opf_case118_ieee", "--from-opf", *outages)
    assert code == 0
    assert "DC cascade" in out


def test_listing_and_info(capsys):
    assert "case118" in run(capsys, "cases")[1].split()
    assert len(run(capsys, "cases", "--pglib")[1].split()) == 66
    assert "radial branches  9" in run(capsys, "info", "pglib_opf_case118_ieee")[1]


def test_errors_are_reported(capsys):
    code, _, err = run(capsys, "pf", "no_such_case")
    assert code == 2
    assert err.startswith("swingbus: error: unknown case")
    with pytest.raises(SystemExit):
        main(["pf"])


def test_module_entry_point():
    completed = subprocess.run(
        [sys.executable, "-m", "swingbus", "--version"], capture_output=True, text=True, check=True
    )
    assert completed.stdout.strip() == f"swingbus {sb.__version__}"
