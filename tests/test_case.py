from __future__ import annotations

import hashlib
import io

import numpy as np
import pytest

import swingbus as sb
from swingbus import data, idx
from swingbus._pglib import PGLIB_FILES, PGLIB_VERSION
from swingbus.case import Case


def small_case() -> Case:
    bus = [[1, 3, 0, 0, 0, 0, 1, 1, 0, 230, 1, 1.1, 0.9], [2, 1, 50, 10, 0, 0, 1, 1, 0, 230, 1, 1.1, 0.9]]
    gen = [[1, 50, 0, 100, -100, 1, 100, 1, 100, 0]]
    branch = [[1, 2, 0.01, 0.1, 0.02, 100, 100, 100, 0, 0, 1, -360, 360]]
    return Case(100, bus, gen, branch, name="small")


def test_validation_errors():
    good = small_case()
    with pytest.raises(ValueError, match="base_mva"):
        Case(0, good.bus, good.gen, good.branch)
    duplicate = good.bus.copy()
    duplicate[1, idx.BUS_I] = 1
    with pytest.raises(ValueError, match="duplicate"):
        Case(100, duplicate, good.gen, good.branch)
    orphan = good.gen.copy()
    orphan[0, idx.GEN_BUS] = 7
    with pytest.raises(ValueError, match="generator bus"):
        Case(100, good.bus, orphan, good.branch)
    dangling = good.branch.copy()
    dangling[0, idx.T_BUS] = 9
    with pytest.raises(ValueError, match="branch to bus"):
        Case(100, good.bus, good.gen, dangling)
    with pytest.raises(ValueError, match="at least one bus"):
        Case(100, np.zeros((0, 13)), np.zeros((0, 21)), np.zeros((0, 13)))
    with pytest.raises(ValueError, match="bus_name"):
        Case(100, good.bus, good.gen, good.branch, bus_name=["only one"])


def test_indexing_helpers(case14):
    assert case14.bus_index([1, 14]).tolist() == [0, 13]
    assert case14.branch_index(2, 1) == 0
    with pytest.raises(KeyError):
        case14.bus_index([99])
    with pytest.raises(KeyError):
        case14.branch_index(1, 14)


def test_copy_is_independent(case14):
    twin = case14.copy()
    twin.bus[0, idx.PD] = 999.0
    assert case14.bus[0, idx.PD] != 999.0
    assert repr(case14) == "Case('case14', buses=14, generators=5, branches=20, base_mva=100)"


def test_load_variants(case14, tmp_path):
    assert sb.load(case14) is not case14
    assert sb.load("case14.m").nb == 14
    path = sb.write_matpower(case14, tmp_path / "mine.m")
    assert sb.load(path).name == "mine"
    assert sb.load(str(path)).nb == 14
    with pytest.raises(KeyError, match="unknown case"):
        sb.load("case_does_not_exist")
    with pytest.raises(FileNotFoundError):
        sb.load(tmp_path / "missing.m")


def test_case_listings():
    bundled = sb.bundled_cases()
    assert bundled[0] == "pglib_opf_case3_lmbd"
    assert {"case14", "case118", "case300", "pglib_opf_case118_ieee"} <= set(bundled)
    assert len(sb.pglib_cases()) == len(PGLIB_FILES) == 66
    assert sb.pglib_cases()[-1] == "pglib_opf_case78484_epigrids"
    assert all(len(digest) == 64 for _, digest in PGLIB_FILES.values())


def test_cache_dir_override(isolated_cache):
    assert sb.cache_dir() == isolated_cache


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_fetch_verifies_checksum(isolated_cache, monkeypatch):
    payload = data.resources.files("swingbus").joinpath("cases", "pglib_opf_case5_pjm.m").read_bytes()
    name = "pglib_opf_case5_pjm"
    assert hashlib.sha256(payload).hexdigest() == PGLIB_FILES[name][1]
    calls = []

    def fake_urlopen(url, timeout):
        calls.append(url)
        return FakeResponse(payload)

    monkeypatch.setattr(data.urllib.request, "urlopen", fake_urlopen)
    path = sb.fetch(name)
    assert path.read_bytes() == payload
    assert calls[0].endswith("/v23.07/pglib_opf_case5_pjm.m")
    sb.fetch(name)
    assert len(calls) == 1
    sb.fetch(name, force=True)
    assert len(calls) == 2


def test_fetch_rejects_tampered_download(isolated_cache, monkeypatch):
    monkeypatch.setattr(data.urllib.request, "urlopen", lambda url, timeout: FakeResponse(b"tampered"))
    with pytest.raises(OSError, match="checksum"):
        sb.fetch("pglib_opf_case1354_pegase")
    assert not any(isolated_cache.rglob("*.m"))
    with pytest.raises(KeyError):
        sb.fetch("not_a_pglib_case")


def test_load_downloads_unbundled_pglib_case(isolated_cache, monkeypatch):
    source = sb.format_matpower(sb.load("pglib_opf_case14_ieee"))
    target = isolated_cache / "pglib-opf-v23.07" / "pglib_opf_case2383wp_k.m"
    monkeypatch.setattr(data, "fetch", lambda name: target)
    target.parent.mkdir(parents=True)
    target.write_text(source)
    assert sb.load("pglib_opf_case2383wp_k").nb == 14


@pytest.mark.network
def test_fetch_real_download(isolated_cache):
    path = sb.fetch("pglib_opf_case89_pegase")
    assert sb.read_matpower(path).nb == 89


@pytest.mark.network
def test_every_pglib_file_is_still_published():
    for name, (size, _) in PGLIB_FILES.items():
        url = data.PGLIB_URL.format(version=PGLIB_VERSION, name=name)
        request = data.urllib.request.Request(url, method="HEAD")
        with data.urllib.request.urlopen(request, timeout=60) as response:
            assert int(response.headers["Content-Length"]) == size, name
