from __future__ import annotations

from typing import NamedTuple

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu, spsolve

from swingbus.case import FloatArray
from swingbus.network import ComplexArray, IntArray, Sparse


class Solution(NamedTuple):
    V: ComplexArray
    converged: bool
    iterations: int
    history: list[float]


def mismatch(Ybus: Sparse, V: ComplexArray, Sbus: ComplexArray) -> ComplexArray:
    return np.asarray(V * np.conj(Ybus @ V) - Sbus)


def _norm(*parts: FloatArray) -> float:
    return max((float(np.max(np.abs(part))) for part in parts if part.size), default=0.0)


def dsbus_dv(Ybus: Sparse, V: ComplexArray) -> tuple[Sparse, Sparse]:
    current = Ybus @ V
    diag_v = sp.diags_array(V)
    diag_i = sp.diags_array(current)
    diag_vnorm = sp.diags_array(V / np.abs(V))
    ds_dvm = diag_v @ (Ybus @ diag_vnorm).conj() + diag_i.conj() @ diag_vnorm
    ds_dva = 1j * diag_v @ (diag_i - Ybus @ diag_v).conj()
    return sp.csr_array(ds_dvm), sp.csr_array(ds_dva)


def jacobian(Ybus: Sparse, V: ComplexArray, pv: IntArray, pq: IntArray) -> Sparse:
    ds_dvm, ds_dva = dsbus_dv(Ybus, V)
    pvpq = np.r_[pv, pq]
    rows_p = ds_dva[pvpq]
    rows_q = ds_dva[pq]
    vm_p = ds_dvm[pvpq]
    vm_q = ds_dvm[pq]
    return sp.block_array(
        [
            [rows_p[:, pvpq].real, vm_p[:, pq].real],
            [rows_q[:, pvpq].imag, vm_q[:, pq].imag],
        ],
        format="csc",
    )


class JacobianPattern:
    def __init__(self, Ybus: Sparse, pv: IntArray, pq: IntArray) -> None:
        nb = Ybus.shape[0]
        coo = sp.coo_array(Ybus)
        diagonal = np.arange(nb)
        Y = sp.coo_array(
            (np.r_[coo.data, np.zeros(nb)], (np.r_[coo.row, diagonal], np.r_[coo.col, diagonal])),
            shape=(nb, nb),
        ).tocsr()
        self.Y = Y
        self.row = np.repeat(np.arange(nb), np.diff(Y.indptr))
        self.col = Y.indices
        self.diagonal = np.flatnonzero(self.row == self.col)
        pvpq = np.r_[pv, pq]
        angle = np.full(nb, -1, dtype=np.intp)
        angle[pvpq] = np.arange(pvpq.size)
        magnitude = np.full(nb, -1, dtype=np.intp)
        magnitude[pq] = pvpq.size + np.arange(pq.size)
        p_row, q_row = angle[self.row], magnitude[self.row]
        a_col, m_col = angle[self.col], magnitude[self.col]
        self.blocks = [
            (p_row >= 0) & (a_col >= 0),
            (p_row >= 0) & (m_col >= 0),
            (q_row >= 0) & (a_col >= 0),
            (q_row >= 0) & (m_col >= 0),
        ]
        rows = np.concatenate(
            [p_row[self.blocks[0]], p_row[self.blocks[1]], q_row[self.blocks[2]], q_row[self.blocks[3]]]
        )
        cols = np.concatenate(
            [a_col[self.blocks[0]], m_col[self.blocks[1]], a_col[self.blocks[2]], m_col[self.blocks[3]]]
        )
        self.size = pvpq.size + pq.size
        self.order = np.lexsort((rows, cols))
        self.indices = rows[self.order]
        self.indptr = np.r_[0, np.cumsum(np.bincount(cols, minlength=self.size))]

    def matrix(self, V: ComplexArray) -> Sparse:
        data = self.Y.data
        current = self.Y @ V
        unit = V / np.abs(V)
        v_row = V[self.row]
        d_angle = -1j * v_row * np.conj(data * V[self.col])
        d_angle[self.diagonal] += 1j * V * np.conj(current)
        d_magnitude = v_row * np.conj(data * unit[self.col])
        d_magnitude[self.diagonal] += np.conj(current) * unit
        values = np.concatenate(
            [
                d_angle[self.blocks[0]].real,
                d_magnitude[self.blocks[1]].real,
                d_angle[self.blocks[2]].imag,
                d_magnitude[self.blocks[3]].imag,
            ]
        )[self.order]
        return sp.csc_array((values, self.indices, self.indptr), shape=(self.size, self.size))


def newton(
    Ybus: Sparse,
    Sbus: ComplexArray,
    V0: ComplexArray,
    ref: IntArray,
    pv: IntArray,
    pq: IntArray,
    tol: float = 1e-8,
    max_iter: int = 10,
) -> Solution:
    pvpq = np.r_[pv, pq]
    nang = pvpq.size
    pattern = JacobianPattern(Ybus, pv, pq)
    V = V0.astype(np.complex128)
    Vm, Va = np.abs(V), np.angle(V)
    mis = mismatch(Ybus, V, Sbus)
    F = np.r_[mis[pvpq].real, mis[pq].imag]
    history = [_norm(F)]
    iterations = 0
    while history[-1] >= tol and iterations < max_iter:
        iterations += 1
        dx = -np.atleast_1d(spsolve(pattern.matrix(V), F))
        Va[pvpq] += dx[:nang]
        Vm[pq] += dx[nang:]
        V = Vm * np.exp(1j * Va)
        Vm, Va = np.abs(V), np.angle(V)
        mis = mismatch(Ybus, V, Sbus)
        F = np.r_[mis[pvpq].real, mis[pq].imag]
        history.append(_norm(F))
        if not np.isfinite(history[-1]):
            break
    return Solution(V, history[-1] < tol, iterations, history)


def fast_decoupled(
    Ybus: Sparse,
    Sbus: ComplexArray,
    V0: ComplexArray,
    ref: IntArray,
    pv: IntArray,
    pq: IntArray,
    Bp: Sparse,
    Bpp: Sparse,
    tol: float = 1e-8,
    max_iter: int = 30,
) -> Solution:
    pvpq = np.r_[pv, pq]
    V = V0.astype(np.complex128)
    Vm, Va = np.abs(V), np.angle(V)

    def residual() -> tuple[FloatArray, FloatArray]:
        mis = mismatch(Ybus, V, Sbus) / Vm
        return mis[pvpq].real, mis[pq].imag

    P, Q = residual()
    history = [_norm(P, Q)]
    if history[-1] < tol:
        return Solution(V, True, 0, history)
    p_solver = splu(sp.csc_array(Bp[pvpq][:, pvpq])) if pvpq.size else None
    q_solver = splu(sp.csc_array(Bpp[pq][:, pq])) if pq.size else None
    iterations = 0
    while iterations < max_iter:
        iterations += 1
        if p_solver is not None:
            Va[pvpq] -= p_solver.solve(P)
            V = Vm * np.exp(1j * Va)
            P, Q = residual()
            if _norm(P, Q) < tol:
                history.append(_norm(P, Q))
                break
        if q_solver is not None:
            Vm[pq] -= q_solver.solve(Q)
            V = Vm * np.exp(1j * Va)
            P, Q = residual()
        history.append(_norm(P, Q))
        if history[-1] < tol or not np.isfinite(history[-1]):
            break
    return Solution(V, history[-1] < tol, iterations, history)


def gauss_seidel(
    Ybus: Sparse,
    Sbus: ComplexArray,
    V0: ComplexArray,
    ref: IntArray,
    pv: IntArray,
    pq: IntArray,
    tol: float = 1e-8,
    max_iter: int = 1000,
) -> Solution:
    Y = sp.csr_array(Ybus)
    indptr, indices, data = Y.indptr, Y.indices, Y.data
    diagonal = Y.diagonal()
    S = Sbus.astype(np.complex128)
    V = V0.astype(np.complex128)
    Vm = np.abs(V)
    pvpq = np.r_[pv, pq]

    def row_current(k: int) -> complex:
        start, stop = indptr[k], indptr[k + 1]
        return complex(data[start:stop] @ V[indices[start:stop]])

    def residual() -> float:
        mis = mismatch(Ybus, V, S)
        return _norm(mis[pvpq].real, mis[pq].imag)

    history = [residual()]
    iterations = 0
    pq_list, pv_list = pq.tolist(), pv.tolist()
    while history[-1] >= tol and iterations < max_iter:
        iterations += 1
        for k in pq_list:
            V[k] += (np.conj(S[k] / V[k]) - row_current(k)) / diagonal[k]
        for k in pv_list:
            S[k] = S[k].real + 1j * (V[k] * np.conj(row_current(k))).imag
            V[k] += (np.conj(S[k] / V[k]) - row_current(k)) / diagonal[k]
        if pv.size:
            V[pv] = Vm[pv] * V[pv] / np.abs(V[pv])
        history.append(residual())
        if not np.isfinite(history[-1]):
            break
    return Solution(V, history[-1] < tol, iterations, history)


def dc_angles(
    Bbus: Sparse, Pbus: FloatArray, Va0: FloatArray, ref: IntArray, pv: IntArray, pq: IntArray
) -> FloatArray:
    pvpq = np.r_[pv, pq]
    Va = Va0.astype(np.float64)
    if pvpq.size:
        rows = Bbus[pvpq]
        rhs = Pbus[pvpq] - rows[:, ref] @ Va0[ref]
        Va[pvpq] = np.atleast_1d(spsolve(sp.csc_array(rows[:, pvpq]), rhs))
    return Va
