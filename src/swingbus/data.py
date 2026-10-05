from __future__ import annotations

import hashlib
import os
import re
import sys
import urllib.request
from importlib import resources
from pathlib import Path
from typing import Union

from swingbus._pglib import PGLIB_FILES, PGLIB_VERSION
from swingbus.case import Case
from swingbus.matpower import parse_matpower, read_matpower

CaseLike = Union[Case, str, "os.PathLike[str]"]
PGLIB_URL = "https://raw.githubusercontent.com/power-grid-lib/pglib-opf/{version}/{name}.m"


def _size_key(name: str) -> tuple[int, str]:
    match = re.search(r"(\d+)", name)
    return (int(match.group(1)) if match else 0, name)


def bundled_cases() -> list[str]:
    folder = resources.files("swingbus") / "cases"
    names = [entry.name[:-2] for entry in folder.iterdir() if entry.name.endswith(".m")]
    return sorted(names, key=_size_key)


def pglib_cases() -> list[str]:
    return sorted(PGLIB_FILES, key=_size_key)


def cache_dir() -> Path:
    override = os.environ.get("SWINGBUS_DATA")
    if override:
        return Path(override)
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Caches"
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return base / "swingbus"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(name: str, *, force: bool = False) -> Path:
    name = name.removesuffix(".m")
    if name not in PGLIB_FILES:
        raise KeyError(f"{name!r} is not a PGLib-OPF {PGLIB_VERSION} case")
    _, expected = PGLIB_FILES[name]
    target = cache_dir() / f"pglib-opf-{PGLIB_VERSION}" / f"{name}.m"
    if target.is_file() and not force and _sha256(target) == expected:
        return target
    url = PGLIB_URL.format(version=PGLIB_VERSION, name=name)
    with urllib.request.urlopen(url, timeout=120) as response:
        payload = response.read()
    if hashlib.sha256(payload).hexdigest() != expected:
        raise OSError(f"checksum mismatch for {url}")
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".part")
    partial.write_bytes(payload)
    partial.replace(target)
    return target


def load(source: CaseLike) -> Case:
    if isinstance(source, Case):
        return source.copy()
    path = Path(os.fspath(source))
    if path.is_file():
        return read_matpower(path)
    name = path.name.removesuffix(".m") if path.suffix == ".m" else os.fspath(source)
    bundled = resources.files("swingbus") / "cases" / f"{name}.m"
    if bundled.is_file():
        return parse_matpower(bundled.read_text(encoding="utf-8"), name=name)
    if name in PGLIB_FILES:
        return read_matpower(fetch(name))
    if path.suffix == ".m":
        raise FileNotFoundError(f"no such case file: {path}")
    raise KeyError(
        f"unknown case {name!r}: use a path to a MATPOWER .m file, one of the bundled cases "
        f"({', '.join(bundled_cases())}) or a PGLib-OPF name such as pglib_opf_case2383wp_k"
    )


def as_case(source: CaseLike) -> Case:
    return source if isinstance(source, Case) else load(source)
