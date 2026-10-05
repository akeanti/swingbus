from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.optimize import linprog

from swingbus.qp import solve_qp


def random_feasible_problem(rng, n, m, p, quadratic):
    A = rng.standard_normal((m, n))
    G = rng.standard_normal((p, n))
    x0 = rng.standard_normal(n)
    b = A @ x0
    h = G @ x0 + rng.uniform(0.1, 1.0, p)
    G = np.vstack([G, np.eye(n), -np.eye(n)])
    h = np.r_[h, np.full(n, 5.0), np.full(n, 5.0)]
    c = rng.standard_normal(n)
    if quadratic:
        M = rng.standard_normal((n, n))
        H = M @ M.T + 0.1 * np.eye(n)
    else:
        H = np.zeros((n, n))
    return H, c, A, b, G, h


@pytest.mark.parametrize("seed", range(6))
def test_linear_programs_match_highs(seed):
    rng = np.random.default_rng(seed)
    _, c, A, b, G, h = random_feasible_problem(rng, 8, 3, 10, quadratic=False)
    ours = solve_qp(None, c, sp.csr_array(A), b, sp.csr_array(G), h)
    reference = linprog(c, A_ub=G, b_ub=h, A_eq=A, b_eq=b, bounds=(None, None), method="highs")
    assert ours.converged
    assert reference.status == 0
    assert ours.objective == pytest.approx(reference.fun, rel=1e-7, abs=1e-7)


@pytest.mark.parametrize("seed", range(6))
def test_quadratic_programs_satisfy_kkt(seed):
    rng = np.random.default_rng(100 + seed)
    H, c, A, b, G, h = random_feasible_problem(rng, 10, 4, 12, quadratic=True)
    result = solve_qp(sp.csr_array(H), c, sp.csr_array(A), b, sp.csr_array(G), h, tol=1e-10)
    x, y, z = result.x, result.eq_dual, result.ineq_dual
    assert result.converged
    np.testing.assert_allclose(H @ x + c + A.T @ y + G.T @ z, 0.0, atol=1e-6)
    np.testing.assert_allclose(A @ x, b, atol=1e-8)
    slack = h - G @ x
    assert slack.min() > -1e-8
    assert z.min() > -1e-8
    assert np.max(np.abs(slack * z)) < 1e-6


def test_equality_only_problem_is_solved_directly():
    H = sp.diags_array([2.0, 4.0])
    c = np.array([-2.0, -8.0])
    A = sp.csr_array(np.array([[1.0, 1.0]]))
    result = solve_qp(H, c, A, np.array([1.0]), sp.csr_array((0, 2)), np.zeros(0))
    assert result.converged
    assert result.iterations == 0
    np.testing.assert_allclose(result.x, [-1.0 / 3.0, 4.0 / 3.0], atol=1e-9)
    np.testing.assert_allclose(result.eq_dual, [8.0 / 3.0], atol=1e-9)


def test_infeasible_problem_does_not_report_success():
    G = sp.csr_array(np.array([[1.0], [-1.0]]))
    h = np.array([-1.0, -1.0])
    result = solve_qp(None, np.array([1.0]), sp.csr_array((0, 1)), np.zeros(0), G, h, max_iter=40)
    assert not result.converged
