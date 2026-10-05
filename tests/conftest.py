from __future__ import annotations

import pytest

import swingbus as sb
from swingbus.case import Case


@pytest.fixture(scope="session")
def case14() -> Case:
    return sb.load("case14")


@pytest.fixture(scope="session")
def case118() -> Case:
    return sb.load("case118")


@pytest.fixture(scope="session")
def pjm5() -> Case:
    return sb.load("pglib_opf_case5_pjm")


@pytest.fixture(scope="session")
def rts24() -> Case:
    return sb.load("pglib_opf_case24_ieee_rts")


@pytest.fixture(scope="session")
def ieee118() -> Case:
    return sb.load("pglib_opf_case118_ieee")


@pytest.fixture
def isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("SWINGBUS_DATA", str(tmp_path))
    return tmp_path
