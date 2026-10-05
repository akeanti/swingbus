import swingbus as sb

case = sb.load("case14")
result = sb.runpf(case)
print(result.report())

print()
for method in ("nr", "fdxb", "fdbx", "gs"):
    solved = sb.runpf(case, method)
    print(f"{solved.method:20s} {solved.iterations:4d} iterations  {solved.elapsed * 1e3:7.2f} ms")
