from __future__ import annotations

import io
import re
from pathlib import Path

import numpy as np

from swingbus.case import Case, FloatArray

_SIGNATURE = re.compile(
    r"^\s*function\s+(?:\[(?P<outputs>[^\]]*)\]|(?P<output>\w+))\s*=\s*(?P<name>\w+)",
    re.MULTILINE,
)
_STRING = re.compile(r"'((?:[^']|'')*)'")
_IDENTIFIER = re.compile(r"[^0-9A-Za-z_]")
_CLOSING = {"[": "]", "{": "}"}


def _code_part(line: str) -> str:
    if "%" not in line:
        return line
    if "'" not in line:
        return line[: line.index("%")]
    quoted = False
    for position, char in enumerate(line):
        if char == "'":
            quoted = not quoted
        elif char == "%" and not quoted:
            return line[:position]
    return line


def _strip_comments(text: str) -> str:
    lines = []
    in_block = False
    for line in text.splitlines():
        marker = line.strip()
        if marker == "%{":
            in_block = True
        elif marker == "%}":
            in_block = False
        elif not in_block:
            lines.append(_code_part(line))
    return "\n".join(lines)


def _matrix(body: str, field: str) -> FloatArray:
    text = body.replace("...", " ").replace(",", " ").replace(";", "\n")
    if not text.strip():
        return np.zeros((0, 0))
    try:
        return np.loadtxt(io.StringIO(text), dtype=np.float64, ndmin=2)
    except ValueError:
        rows = [line.split() for line in text.splitlines() if line.strip()]
        widths = sorted({len(row) for row in rows})
        if len(widths) > 1:
            raise ValueError(f"rows of mpc.{field} have different lengths: {widths}") from None
        raise ValueError(f"mpc.{field} contains values that are not numbers") from None


def _cell(body: str) -> list[str]:
    return [match.replace("''", "'") for match in _STRING.findall(body)]


def _assignments(text: str, prefix: str) -> dict[str, str]:
    pattern = re.compile(rf"(?<![\w.]){re.escape(prefix)}(\w+)\s*=\s*")
    found: dict[str, str] = {}
    position = 0
    while match := pattern.search(text, position):
        field = match.group(1)
        start = match.end()
        opening = text[start] if start < len(text) else ""
        if opening in _CLOSING:
            end = text.find(_CLOSING[opening], start)
            if end < 0:
                raise ValueError(f"unterminated value for {prefix}{field}")
            found[field] = text[start : end + 1]
            position = end + 1
        else:
            terminator = re.search(r"[;\n]", text[start:])
            stop = start + terminator.start() if terminator else len(text)
            found[field] = text[start:stop].strip()
            position = stop
    return found


def parse_matpower(text: str, name: str | None = None) -> Case:
    code = _strip_comments(text)
    signature = _SIGNATURE.search(code)
    if signature and signature.group("output"):
        prefix = signature.group("output") + "."
    elif signature and signature.group("outputs"):
        prefix = ""
    else:
        prefix = "mpc."
    values = _assignments(code, prefix)
    missing = [field for field in ("baseMVA", "bus", "gen", "branch") if field not in values]
    if missing:
        raise ValueError(f"not a MATPOWER case: missing {', '.join(prefix + m for m in missing)}")
    version = values.get("version", "'2'").strip("'\"")
    if version not in {"1", "2"}:
        raise ValueError(f"unsupported MATPOWER case format version {version!r}")
    base_mva = float(values["baseMVA"].strip("[]"))
    gencost = _matrix(values["gencost"][1:-1], "gencost") if "gencost" in values else None
    bus_name = _cell(values["bus_name"]) if "bus_name" in values else None
    case_name = name or (signature.group("name") if signature else "case")
    return Case(
        base_mva,
        _matrix(values["bus"][1:-1], "bus"),
        _matrix(values["gen"][1:-1], "gen"),
        _matrix(values["branch"][1:-1], "branch"),
        gencost,
        name=case_name,
        bus_name=bus_name or None,
    )


def read_matpower(path: str | Path) -> Case:
    path = Path(path)
    return parse_matpower(path.read_text(encoding="utf-8", errors="replace"), name=path.stem)


def _number(value: float) -> str:
    if np.isnan(value):
        return "NaN"
    if np.isinf(value):
        return "Inf" if value > 0 else "-Inf"
    if value == int(value) and abs(value) < 1e15:
        return str(int(value))
    return repr(float(value))


def _function_name(name: str) -> str:
    cleaned = _IDENTIFIER.sub("_", name) or "case"
    return cleaned if cleaned[0].isalpha() else f"case_{cleaned}"


def format_matpower(case: Case, name: str | None = None) -> str:
    lines = [
        f"function mpc = {_function_name(name or case.name)}",
        "mpc.version = '2';",
        f"mpc.baseMVA = {_number(case.base_mva)};",
    ]
    tables: list[tuple[str, FloatArray | None]] = [
        ("bus", case.bus),
        ("gen", case.gen),
        ("branch", case.branch),
        ("gencost", case.gencost),
    ]
    for field, table in tables:
        if table is None:
            continue
        lines.append(f"mpc.{field} = [")
        lines.extend("\t" + "\t".join(_number(value) for value in row) + ";" for row in table)
        lines.append("];")
    if case.bus_name:
        lines.append("mpc.bus_name = {")
        lines.extend("\t'" + label.replace("'", "''") + "';" for label in case.bus_name)
        lines.append("};")
    return "\n".join(lines) + "\n"


def write_matpower(case: Case, path: str | Path) -> Path:
    path = Path(path)
    path.write_text(format_matpower(case, name=path.stem), encoding="utf-8")
    return path
