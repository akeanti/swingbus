from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from numpy.typing import ArrayLike
from scipy.sparse.linalg import spsolve

from swingbus import idx
from swingbus.case import FloatArray
from swingbus.data import CaseLike, as_case
from swingbus.network import IslandingError, bus_types, islands, make_bdc, to_internal

ISLANDING_TOLERANCE = 1e-8


def ptdf(case: CaseLike, slack: int | ArrayLike | None = None) -> FloatArray:
    source = as_case(case)
    work, index = to_internal(source)
    if np.unique(islands(work)).size > 1:
        raise IslandingError(f"{source.name} is not connected; PTDF needs a single island")
    Bbus, Bf, _, _ = make_bdc(work)
    reference = int(bus_types(work)[0][0])
    weights = None
    if slack is None:
        anchor = reference
    elif np.ndim(slack) == 0:
        bus_number = int(np.asarray(slack))
        position = np.flatnonzero(index.bus_rows == source.bus_index(bus_number)[0])
        if position.size == 0:
            raise ValueError(f"slack bus {bus_number} is not energized")
        anchor = int(position[0])
    else:
        raw = np.asarray(slack, dtype=np.float64)
        if raw.shape != (source.nb,) or raw.sum() <= 0:
            raise ValueError("distributed slack weights need one non-negative entry per bus")
        weights = raw[index.bus_rows] / raw[index.bus_rows].sum()
        anchor = reference
    keep = np.delete(np.arange(work.nb), anchor)
    H = np.zeros((work.nl, work.nb))
    if keep.size and work.nl:
        rhs = Bf[:, keep].T.toarray()
        solved = np.asarray(spsolve(sp.csc_array(Bbus[keep][:, keep]), rhs))
        H[:, keep] = solved.reshape(keep.size, -1).T
    if weights is not None:
        H -= (H @ weights)[:, None]
    full = np.zeros((source.nl, source.nb))
    full[np.ix_(index.branch_rows, index.bus_rows)] = H
    return full


def lodf(case: CaseLike, H: FloatArray | None = None) -> FloatArray:
    source = as_case(case)
    H = ptdf(source) if H is None else np.asarray(H, dtype=np.float64)
    in_service = np.zeros(source.nl, dtype=bool)
    in_service[to_internal(source)[1].branch_rows] = True
    f = source.bus_index(source.branch[:, idx.F_BUS])
    t = source.bus_index(source.branch[:, idx.T_BUS])
    transfer = H[:, f] - H[:, t]
    denominator = 1.0 - np.diag(transfer)
    islanding = in_service & (np.abs(denominator) < ISLANDING_TOLERANCE)
    safe = np.where(islanding | ~in_service, 1.0, denominator)
    L = transfer / safe[None, :]
    L[:, islanding] = np.nan
    L[:, ~in_service] = 0.0
    L[~in_service, :] = 0.0
    diagonal = np.flatnonzero(in_service & ~islanding)
    L[diagonal, diagonal] = -1.0
    return L
