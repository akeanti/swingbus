import swingbus as sb
from swingbus import idx

case = sb.load("pglib_opf_case5_pjm")
result = sb.rundcopf(case)
print(result)

print()
print("bus   LMP ($/MWh)")
for number, price in zip(case.bus[:, idx.BUS_I].astype(int), result.lmp, strict=True):
    print(f"{number:3d}   {price:10.4f}")

relaxed = case.copy()
relaxed.branch[:, idx.RATE_A] = 0
flat = sb.rundcopf(relaxed)
print()
print(f"with unlimited lines: {flat.objective:.2f} $/h, single price {flat.lmp[0]:.4f} $/MWh")
print(f"cost of congestion:   {result.objective - flat.objective:.2f} $/h")
