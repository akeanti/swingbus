import platform
import sys
import time

import numpy as np
import scipy

import swingbus as sb

CASES = [
    "case14",
    "case118",
    "case300",
    "pglib_opf_case1354_pegase",
    "pglib_opf_case2383wp_k",
    "pglib_opf_case9241_pegase",
]


def best(function, repeat):
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        function()
        times.append(time.perf_counter() - start)
    return 1e3 * min(times)


def main():
    print(f"python {platform.python_version()}, numpy {np.__version__}, scipy {scipy.__version__}")
    print(f"{platform.processor() or platform.machine()} on {platform.system()} {platform.release()}")
    print()
    print("| case | buses | branches | NR iterations | NR (ms) | DC PF (ms) | DC-OPF (ms) |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for name in CASES:
        case = sb.load(name)
        init = "case" if name.startswith("case") else "flat"
        sb.runpf(case, init=init)
        result = sb.runpf(case, init=init)
        nr = best(lambda case=case, init=init: sb.runpf(case, init=init), 7)
        dc = best(lambda case=case: sb.rundcpf(case), 7)
        opf = best(lambda case=case: sb.rundcopf(case), 3 if case.nb < 2000 else 1)
        status = str(result.iterations) if result.converged else "did not converge"
        print(f"| `{name}` | {case.nb} | {case.nl} | {status} | {nr:.1f} | {dc:.1f} | {opf:.0f} |")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
