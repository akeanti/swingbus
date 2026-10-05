from __future__ import annotations

import numpy as np
import pytest

import swingbus as sb
from swingbus import plot

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")


@pytest.fixture(autouse=True)
def close_figures():
    yield
    import matplotlib.pyplot as plt

    plt.close("all")


def test_layout_is_deterministic_and_normalised(case14):
    first = plot.layout(case14)
    second = plot.layout(case14)
    np.testing.assert_array_equal(first, second)
    assert first.shape == (14, 2)
    assert first.min() >= 0.0
    assert first.max() == pytest.approx(1.0)


def test_layout_keeps_neighbours_close(case118):
    positions = plot.layout(case118)
    on = case118.branch[:, 10] > 0
    f = case118.bus_index(case118.branch[on, 0])
    t = case118.bus_index(case118.branch[on, 1])
    neighbours = np.linalg.norm(positions[f] - positions[t], axis=1).mean()
    rng = np.random.default_rng(0)
    pairs = rng.integers(0, case118.nb, size=(500, 2))
    strangers = np.linalg.norm(positions[pairs[:, 0]] - positions[pairs[:, 1]], axis=1).mean()
    assert neighbours < 0.5 * strangers


def test_plots_render(case118, pjm5):
    flow = sb.runpf(case118)
    assert plot.plot_network(flow).get_title() == "case118"
    assert plot.plot_network(sb.rundcopf(pjm5), title="prices").get_title() == "prices"
    assert plot.plot_network(case118, bus_values="none") is not None
    assert plot.plot_voltage(flow).get_ylabel() == "voltage magnitude (pu)"
    solvers = [sb.runpf(case118, method) for method in ("nr", "fdxb")]
    assert len(plot.plot_convergence(solvers).get_lines()) == 2


def test_tiny_and_huge_layouts(case14):
    tiny = sb.Case(100, case14.bus[:2], case14.gen[:1], case14.branch[:1])
    assert plot.layout(tiny).shape == (2, 2)
    buses = np.tile(case14.bus[:1], (3001, 1))
    buses[:, 0] = np.arange(1, 3002)
    huge = sb.Case(100, buses, case14.gen[:0], case14.branch[:0])
    with pytest.raises(ValueError, match="limited"):
        plot.layout(huge)
