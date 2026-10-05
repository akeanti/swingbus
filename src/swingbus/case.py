from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from swingbus import idx

FloatArray = NDArray[np.float64]


def column(table: FloatArray, index: int) -> FloatArray:
    return np.array(table[:, index], dtype=np.float64)


def pad_columns(table: FloatArray, width: int) -> FloatArray:
    if table.shape[1] >= width:
        return table
    padded = np.zeros((table.shape[0], width))
    padded[:, : table.shape[1]] = table
    return padded


def _table(data: ArrayLike, width: int, label: str) -> FloatArray:
    table = np.array(data, dtype=np.float64, ndmin=2)
    if table.ndim != 2:
        raise ValueError(f"{label} must be a 2-D table, got shape {table.shape}")
    if table.size == 0:
        return np.zeros((0, width))
    return pad_columns(table, width)


@dataclass(eq=False)
class Case:
    base_mva: float
    bus: FloatArray
    gen: FloatArray
    branch: FloatArray
    gencost: FloatArray | None = None
    name: str = "case"
    bus_name: list[str] | None = None

    def __post_init__(self) -> None:
        self.base_mva = float(self.base_mva)
        if not self.base_mva > 0:
            raise ValueError(f"base_mva must be positive, got {self.base_mva}")
        self.bus = _table(self.bus, idx.BUS_COLS, "bus")
        self.gen = _table(self.gen, idx.GEN_COLS, "gen")
        branch_cols = np.array(self.branch, dtype=np.float64, ndmin=2).shape[1]
        self.branch = _table(self.branch, idx.BRANCH_COLS, "branch")
        if 0 < branch_cols <= idx.ANGMIN:
            self.branch[:, idx.ANGMIN] = -360.0
            self.branch[:, idx.ANGMAX] = 360.0
        if self.gencost is not None:
            self.gencost = np.array(self.gencost, dtype=np.float64, ndmin=2)
            if self.gencost.size == 0:
                self.gencost = None
        if self.bus_name is not None:
            self.bus_name = [str(item) for item in self.bus_name]
            if len(self.bus_name) != self.nb:
                raise ValueError(f"bus_name has {len(self.bus_name)} entries for {self.nb} buses")
        self._validate()

    def _validate(self) -> None:
        if self.nb == 0:
            raise ValueError("a case needs at least one bus")
        numbers = self.bus[:, idx.BUS_I]
        if np.any(numbers < 0) or np.any(numbers != np.round(numbers)):
            raise ValueError("bus numbers must be non-negative integers")
        unique, counts = np.unique(numbers, return_counts=True)
        if np.any(counts > 1):
            raise ValueError(f"duplicate bus numbers: {unique[counts > 1].astype(int).tolist()}")
        for label, refs in (
            ("generator", self.gen[:, idx.GEN_BUS]),
            ("branch from", self.branch[:, idx.F_BUS]),
            ("branch to", self.branch[:, idx.T_BUS]),
        ):
            missing = np.setdiff1d(refs, unique)
            if missing.size:
                raise ValueError(f"{label} bus not in bus table: {missing.astype(int).tolist()}")

    @property
    def nb(self) -> int:
        return int(self.bus.shape[0])

    @property
    def ng(self) -> int:
        return int(self.gen.shape[0])

    @property
    def nl(self) -> int:
        return int(self.branch.shape[0])

    def bus_index(self, numbers: ArrayLike) -> NDArray[np.intp]:
        lookup = dict(zip(self.bus[:, idx.BUS_I].astype(np.int64).tolist(), range(self.nb), strict=True))
        values = np.atleast_1d(np.asarray(numbers)).astype(np.int64)
        try:
            return np.array([lookup[int(value)] for value in values], dtype=np.intp)
        except KeyError as error:
            raise KeyError(f"bus {error.args[0]} is not in {self.name}") from None

    def branch_index(self, from_bus: int, to_bus: int) -> int:
        f = self.branch[:, idx.F_BUS]
        t = self.branch[:, idx.T_BUS]
        hits = np.flatnonzero(((f == from_bus) & (t == to_bus)) | ((f == to_bus) & (t == from_bus)))
        if hits.size == 0:
            raise KeyError(f"no branch between buses {from_bus} and {to_bus} in {self.name}")
        return int(hits[0])

    def copy(self) -> Case:
        return Case(
            self.base_mva,
            self.bus.copy(),
            self.gen.copy(),
            self.branch.copy(),
            None if self.gencost is None else self.gencost.copy(),
            name=self.name,
            bus_name=None if self.bus_name is None else list(self.bus_name),
        )

    def __repr__(self) -> str:
        return (
            f"Case({self.name!r}, buses={self.nb}, generators={self.ng}, "
            f"branches={self.nl}, base_mva={self.base_mva:g})"
        )
