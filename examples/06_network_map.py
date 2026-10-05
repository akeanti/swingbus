from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import swingbus as sb
from swingbus.plot import layout, plot_network

case = sb.load("pglib_opf_case118_ieee")
positions = layout(case)
market = sb.rundcopf(case)
physics = sb.runpf(market.case, init="flat")

figure, (left, right) = plt.subplots(1, 2, figsize=(15, 7.6))
plot_network(market, ax=left, positions=positions, title="DC-OPF prices")
plot_network(physics, ax=right, positions=positions, title="AC power flow of the same dispatch")
figure.suptitle("IEEE 118-bus system, PGLib-OPF data", fontsize=14)

target = Path(__file__).resolve().parents[1] / "docs" / "images" / "ieee118.png"
target.parent.mkdir(parents=True, exist_ok=True)
figure.savefig(target, dpi=150, bbox_inches="tight")
print(f"saved {target}")
print(market.summary().splitlines()[0])
print(physics.summary().splitlines()[0])
