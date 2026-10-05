import numpy as np

import swingbus as sb

case = sb.load("pglib_opf_case118_ieee")
dispatch = sb.rundcopf(case)
print(dispatch.summary().splitlines()[0])

screen = sb.n1(dispatch.case)
print()
print(screen.table(top=8))

suspects = screen.outages[screen.ranking()[:20]]
suspects = suspects[~np.isin(suspects, screen.outages[screen.islanding])]
detailed = sb.n1(dispatch.case, method="ac", branches=suspects)
print()
print(detailed.table(top=8))
