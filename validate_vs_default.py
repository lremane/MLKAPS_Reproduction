#!/usr/bin/env python3

import csv
import math
from kernel import reformulate, run_driver

TREE_RESULTS_CSV = "mlkaps_output/optim.csv"
OUT_CSV = "validation_vs_default.csv"

# ScaLAPACK's recommended default distribution parameters for PDGEQRF:
# - MB = NB = 64
# - QR prefers a squared grid (P = Q)
DEFAULT_MB = 64
DEFAULT_NB = 64
DEFAULT_P = 11
DEFAULT_Q = 11
DEFAULT_NPROC = DEFAULT_P * DEFAULT_Q
DEFAULT_NTHREADS = 1

rows = []

with open(TREE_RESULTS_CSV, newline="") as f, open(OUT_CSV, "a", newline="") as out_f:
    reader = csv.DictReader(f)
    writer = None

    for row in reader:
        m = int(float(row["m"]))
        n = int(float(row["n"]))

        alpha = float(row["alpha"])
        gamma = float(row["gamma"])
        beta = float(row["beta"])
        p_frac = float(row["p_frac"])

        mb, nb, p, q, npernode, nproc, nthreads = reformulate(
            m, n, alpha, beta, gamma, p_frac
        )

        tree_time = run_driver(
            m, n, mb, nb, p, q, nproc, nthreads
        )

        default_time = run_driver(
            m, n,
            DEFAULT_MB, DEFAULT_NB,
            DEFAULT_P, DEFAULT_Q,
            DEFAULT_NPROC, DEFAULT_NTHREADS
        )

        if tree_time is not None and default_time is not None and tree_time > 0:
            speedup = default_time / tree_time
        else:
            speedup = None

        print(
            f"m={m} n={n} "
            f"tree={tree_time} default={default_time} speedup={speedup}"
        )

        rows.append({
            "m": m,
            "n": n,
            "tree_mb": mb,
            "tree_nb": nb,
            "tree_p": p,
            "tree_q": q,
            "tree_nproc": nproc,
            "tree_time": tree_time,
            "default_mb": DEFAULT_MB,
            "default_nb": DEFAULT_NB,
            "default_p": DEFAULT_P,
            "default_q": DEFAULT_Q,
            "default_nproc": DEFAULT_NPROC,
            "default_time": default_time,
            "speedup_default_over_tree": speedup,
        })

        if writer is None:
            writer = csv.DictWriter(out_f, fieldnames=rows[0].keys())
            writer.writeheader()

        writer.writerow(rows[-1])
        out_f.flush()

valid = [
    r["speedup_default_over_tree"]
    for r in rows
    if r["speedup_default_over_tree"] is not None
]

tree_times = [
    r["tree_time"]
    for r in rows
    if r["tree_time"] is not None
]

default_times = [
    r["default_time"]
    for r in rows
    if r["default_time"] is not None
]

mean_tree = sum(tree_times) / len(tree_times)
mean_default = sum(default_times) / len(default_times)
mean_speedup = sum(valid) / len(valid)
geomean_speedup = math.exp(sum(math.log(x) for x in valid if x > 0) / len(valid))
fraction_faster = sum(1 for x in valid if x > 1.0) / len(valid)

print()
print("Summary")
print("-------")
print(f"points: {len(valid)}")
print(f"mean tree time: {mean_tree}")
print(f"mean default time: {mean_default}")
print(f"mean speedup default/tree: {mean_speedup}")
print(f"geomean speedup default/tree: {geomean_speedup}")
print(f"fraction where MLKAPS is faster: {fraction_faster}")
print(f"wrote {OUT_CSV}")