from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

from swingbus.case import FloatArray
from swingbus.network import Sparse

FRACTION_TO_BOUNDARY = 0.995
REGULARIZATION = 1e-10
DIVERGED = 1e30


@dataclass(frozen=True)
class QPResult:
    x: FloatArray
    eq_dual: FloatArray
    ineq_dual: FloatArray
    objective: float
    converged: bool
    iterations: int


class _Problem(NamedTuple):
    H: Sparse
    c: FloatArray
    A: Sparse
    b: FloatArray
    G: Sparse
    h: FloatArray


class _Point(NamedTuple):
    x: FloatArray
    y: FloatArray
    z: FloatArray
    s: FloatArray


def _norm(vector: FloatArray) -> float:
    return float(np.max(np.abs(vector))) if vector.size else 0.0


def _max_step(value: FloatArray, change: FloatArray) -> float:
    shrinking = change < 0
    if not shrinking.any():
        return np.inf
    return float(np.min(-value[shrinking] / change[shrinking]))


def _factor(problem: _Problem, weights: FloatArray) -> Any:
    n, m = problem.c.size, problem.b.size
    top = problem.H + problem.G.T @ sp.diags_array(weights) @ problem.G
    top = top + REGULARIZATION * sp.eye_array(n)
    lower = -REGULARIZATION * sp.eye_array(m)
    return splu(sp.block_array([[top, problem.A.T], [problem.A, lower]], format="csc"))


def _residuals(problem: _Problem, point: _Point) -> tuple[FloatArray, FloatArray, FloatArray]:
    H, c, A, b, G, h = problem
    x, y, z, s = point
    return H @ x + c + A.T @ y + G.T @ z, A @ x - b, G @ x + s - h


def _direction(
    lu: Any,
    problem: _Problem,
    point: _Point,
    residuals: tuple[FloatArray, FloatArray, FloatArray],
    target: FloatArray,
) -> _Point:
    r_d, r_p, r_i = residuals
    z, s = point.z, point.s
    rhs = np.r_[-r_d - problem.G.T @ ((z * r_i - target) / s), -r_p]
    solved = lu.solve(rhs)
    n = problem.c.size
    dx, dy = solved[:n], solved[n:]
    ds = -r_i - problem.G @ dx
    dz = (-target - z * ds) / s
    return _Point(dx, dy, dz, ds)


def _start(problem: _Problem) -> _Point:
    lu = _factor(problem, np.ones(problem.h.size))
    solved = lu.solve(np.r_[-problem.c + problem.G.T @ problem.h, problem.b])
    n = problem.c.size
    x, y = solved[:n], solved[n:]
    s = problem.h - problem.G @ x
    z = -s
    if s.size and np.min(s) <= 0:
        s = s + 1.0 - np.min(s)
    if z.size and np.min(z) <= 0:
        z = z + 1.0 - np.min(z)
    return _Point(x, y, z, s)


def _predictor_corrector(
    problem: _Problem,
    point: _Point,
    residuals: tuple[FloatArray, FloatArray, FloatArray],
    mu: float,
) -> _Point:
    p = point.s.size
    lu = _factor(problem, point.z / point.s)
    affine = _direction(lu, problem, point, residuals, point.s * point.z)
    alpha = min(1.0, _max_step(point.s, affine.s), _max_step(point.z, affine.z))
    gap = float((point.s + alpha * affine.s) @ (point.z + alpha * affine.z)) / p
    sigma = (gap / mu) ** 3
    target = point.s * point.z + affine.s * affine.z - sigma * mu
    step = _direction(lu, problem, point, residuals, target)
    limit = min(_max_step(point.s, step.s), _max_step(point.z, step.z))
    alpha = min(1.0, FRACTION_TO_BOUNDARY * limit)
    return _Point(*(value + alpha * delta for value, delta in zip(point, step, strict=True)))


def solve_qp(
    H: Sparse | None,
    c: FloatArray,
    A: Sparse,
    b: FloatArray,
    G: Sparse,
    h: FloatArray,
    *,
    tol: float = 1e-9,
    max_iter: int = 100,
) -> QPResult:
    n = c.size
    Hq = sp.csr_array((n, n)) if H is None else sp.csr_array(H)
    scale = max(1.0, _norm(c), _norm(Hq.data))
    problem = _Problem(
        Hq / scale,
        c / scale,
        sp.csr_array(A, shape=(b.size, n)),
        b,
        sp.csr_array(G, shape=(h.size, n)),
        h,
    )
    point = _start(problem)
    p = h.size
    limits = (
        tol * (1.0 + _norm(problem.c)),
        tol * (1.0 + _norm(b)),
        tol * (1.0 + _norm(h)),
    )
    converged = p == 0
    iterations = 0
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        while not converged and iterations < max_iter:
            residuals = _residuals(problem, point)
            mu = float(point.s @ point.z) / p
            if not np.isfinite(mu) or mu > DIVERGED or not np.all(np.isfinite(point.x)):
                break
            if mu <= tol and all(_norm(r) <= limit for r, limit in zip(residuals, limits, strict=True)):
                converged = True
                break
            iterations += 1
            point = _predictor_corrector(problem, point, residuals, mu)
    x = point.x
    return QPResult(
        x,
        point.y * scale,
        point.z * scale,
        float(0.5 * x @ (Hq @ x) + c @ x),
        converged,
        iterations,
    )
