import sys
from pathlib import Path

import numpy as np

import swingbus as sb
from swingbus import idx

count = int(sys.argv[1]) if len(sys.argv) > 1 else 200
case = sb.load("pglib_opf_case118_ieee")
rows = case.bus_index

edge_index = np.stack([rows(case.branch[:, idx.F_BUS]), rows(case.branch[:, idx.T_BUS])])
node_features, edge_features, not_served, cascaded, tripped = [], [], [], [], []

for sample in sb.sample_cascades(case, count, k=2, load_range=(0.85, 1.15), seed=7):
    start = sample.meta["load_scale"]
    pd = case.bus[:, idx.PD] * start
    initial = np.zeros(case.nl)
    initial[sample.initial] = 1.0
    lost = np.zeros(case.nl)
    lost[sample.tripped] = 1.0
    node_features.append(np.column_stack([pd, case.bus[:, idx.BS], np.full(case.nb, start)]))
    edge_features.append(np.column_stack([case.branch[:, idx.BR_X], case.branch[:, idx.RATE_A], initial]))
    not_served.append(sample.shed)
    cascaded.append(sample.cascaded)
    tripped.append(lost)

target = Path("cascades_case118.npz")
np.savez_compressed(
    target,
    edge_index=edge_index,
    x=np.array(node_features),
    edge_attr=np.array(edge_features),
    dns=np.array(not_served),
    cascade=np.array(cascaded),
    tripped=np.array(tripped),
)
shed = np.array(not_served)
print(f"{count} scenarios written to {target}")
print(f"cascades: {np.mean(cascaded):.1%}  demand not served > 0: {np.mean(shed > 1e-6):.1%}")
print(f"mean demand not served: {shed.mean():.1f} MW, worst: {shed.max():.1f} MW")
