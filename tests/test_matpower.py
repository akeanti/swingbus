from __future__ import annotations

import numpy as np
import pytest

import swingbus as sb
from swingbus import idx

MINIMAL = """
function mpc = tiny
mpc.version = '2';
mpc.baseMVA = 100;
mpc.bus = [
    1 3 0 0 0 0 1 1 0 230 1 1.1 0.9;
    2 1 50 10 0 0 1 1 0 230 1 1.1 0.9;
];
mpc.gen = [1 50 0 100 -100 1 100 1 100 0];
mpc.branch = [1 2 0.01 0.1 0.02 100 100 100 0 0 1];
"""


@pytest.mark.parametrize("name", sb.bundled_cases())
def test_bundled_cases_parse(name):
    case = sb.load(name)
    assert case.nb > 0
    assert case.bus.shape[1] >= idx.BUS_COLS
    assert case.gen.shape[1] >= idx.GEN_COLS
    assert case.branch.shape[1] >= idx.BRANCH_COLS
    assert case.gencost is not None
    assert case.gencost.shape[0] >= case.ng


@pytest.mark.parametrize("name", ["case14", "case118", "pglib_opf_case24_ieee_rts", "case300"])
def test_round_trip(name, tmp_path):
    case = sb.load(name)
    path = sb.write_matpower(case, tmp_path / "copy.m")
    again = sb.read_matpower(path)
    assert again.name == "copy"
    assert again.base_mva == case.base_mva
    np.testing.assert_array_equal(again.bus, case.bus)
    np.testing.assert_array_equal(again.gen, case.gen)
    np.testing.assert_array_equal(again.branch, case.branch)
    np.testing.assert_array_equal(again.gencost, case.gencost)
    assert again.bus_name == case.bus_name


def test_minimal_case_pads_columns():
    case = sb.parse_matpower(MINIMAL)
    assert case.name == "tiny"
    assert case.gen.shape == (1, idx.GEN_COLS)
    assert case.branch[0, idx.ANGMIN] == -360
    assert case.branch[0, idx.ANGMAX] == 360
    assert case.gencost is None


def test_comments_strings_and_separators():
    text = """
    %{
    mpc.bus = [ this block is ignored ];
    %}
    function mpc = commented % trailing comment
    mpc.version = '2';
    mpc.baseMVA = 100; % base
    mpc.bus = [
        1, 3, 0, 0, 0, 0, 1, 1, 0, 230, 1, 1.1, 0.9; % slack
        2, 1, 5e1, 10, 0, 0, 1, 1, 0, 230, 1, 1.1, 0.9
    ];
    mpc.gen = [1 50 0 Inf -Inf 1 100 1 100 0];
    mpc.branch = [1 2 0.01 0.1 0.02 100 100 100 0 0 1 -360 360];
    mpc.bus_name = { 'North % yard'; 'O''Brien' };
    """
    case = sb.parse_matpower(text)
    assert case.bus[1, idx.PD] == 50
    assert np.isinf(case.gen[0, idx.QMAX])
    assert case.bus_name == ["North % yard", "O'Brien"]


def test_version_one_format():
    text = """
    function [baseMVA, bus, gen, branch] = old
    baseMVA = 100;
    bus = [1 3 0 0 0 0 1 1 0 230 1 1.1 0.9; 2 1 50 10 0 0 1 1 0 230 1 1.1 0.9];
    gen = [1 50 0 100 -100 1 100 1 100 0];
    branch = [1 2 0.01 0.1 0.02 100 100 100 0 0 1];
    """
    case = sb.parse_matpower(text)
    assert case.nb == 2
    assert case.name == "old"


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("mpc.baseMVA = 100;", "missing"),
        (MINIMAL.replace("1 3 0 0 0 0 1 1 0 230 1 1.1 0.9;", "1 3 0 0;"), "different lengths"),
        (MINIMAL.replace("mpc.version = '2';", "mpc.version = '3';"), "version"),
        (MINIMAL.replace("230 1 1.1 0.9;\n];", "x 1 1.1 0.9;\n];"), "not numbers"),
    ],
)
def test_malformed_input(text, message):
    with pytest.raises(ValueError, match=message):
        sb.parse_matpower(text)


def test_format_uses_valid_function_name():
    case = sb.parse_matpower(MINIMAL)
    text = sb.format_matpower(case, name="9 bus-variant")
    assert text.startswith("function mpc = case_9_bus_variant")
    assert sb.parse_matpower(text).nb == 2


def test_format_special_values():
    case = sb.parse_matpower(MINIMAL)
    case.gen[0, idx.QMAX] = np.inf
    case.gen[0, idx.QMIN] = -np.inf
    case.bus[0, idx.VA] = np.nan
    again = sb.parse_matpower(sb.format_matpower(case))
    assert np.isposinf(again.gen[0, idx.QMAX])
    assert np.isneginf(again.gen[0, idx.QMIN])
    assert np.isnan(again.bus[0, idx.VA])
