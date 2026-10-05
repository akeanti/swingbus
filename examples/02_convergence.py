from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import swingbus as sb
from swingbus.plot import plot_convergence

case = sb.load("case_ieee30")
results = [sb.runpf(case, method, init="flat") for method in ("nr", "fdxb", "fdbx", "gs")]

figure, ax = plt.subplots(figsize=(7, 4.2))
plot_convergence(results, ax=ax)
ax.set_xscale("symlog", linthresh=10)
ax.set_title("IEEE 30-bus from a flat start")
figure.tight_layout()

target = Path(__file__).resolve().parents[1] / "docs" / "images" / "convergence.png"
target.parent.mkdir(parents=True, exist_ok=True)
figure.savefig(target, dpi=160)
print(f"saved {target}")
for result in results:
    print(f"{result.method:20s} {result.iterations:4d} iterations")
