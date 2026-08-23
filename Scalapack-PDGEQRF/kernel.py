#!/usr/bin/env python3
"""
MLKAPS wrapper for ScaLAPACK PDGEQRF on JURECA (single node, up to 128 cores).

MLKAPS cannot do constrained optimization, so every design parameter it sees is
a FREE parameter in [0,1]. This wrapper maps those to always-feasible ScaLAPACK
parameters using the Table 5 lerp reformulation (bunit=8, dependency order
p -> npernode -> mb,nb), exactly as the paper did. The large PENALTY is only a
safety net for runtime failures, never for infeasibility.
"""

import sys
import os
import subprocess
import tempfile
import shutil
import math

# Cluster constants (single node, up to 128 cores)
NODES = 1
CORES = 128
BUNIT = 8            # block unit, the "8" in the Table 5 formulas
BLOCK_UNIT_CAP = 16  # the "16" cap in the Table 5 formulas
NITER = 3
PENALTY = 1e12

DRIVER = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "scalapack-driver", "bin", "jsc", "pdqrdriver",
)
SRUN_FLAGS = ["--exact", "--overlap", "--mpi=pspmix"]  # confirmed via standalone test


def lerp(t, lb, ub):
    """Continuous linear interpolation, clamped."""
    if ub < lb:
        ub = lb
    t = max(0.0, min(1.0, t))
    return lb + t * (ub - lb)


def reformulate(m, n, alpha, beta, gamma, p_frac):
    """
    Map free params in [0,1] to a feasible, self-consistent config.
    Single node: nproc == npernode. Grid p*q == nproc by construction.
    Order: nproc -> p (a divisor of nproc) -> q -> block sizes.
    """
    # 1) total processes: free param beta picks nproc in [1, CORES]
    nproc = int(round(lerp(beta, 1, CORES)))
    nproc = max(1, min(CORES, nproc))

    # 2) p must divide nproc so the grid is exact. Pick p among divisors.
    divs = [d for d in range(1, nproc + 1) if nproc % d == 0]
    p = divs[int(round(lerp(p_frac, 0, len(divs) - 1)))]
    q = nproc // p                      # exact: p*q == nproc

    # todo: only works singled threaded
    npernode = nproc                    # single node
    #nthreads = max(1, CORES // npernode)
    nthreads = 1                        # pin to 1 for now (avoid cpus-per-task conflict)

    # 3) block sizes via lerp, feasible by the cst1/cst2 bounds. Follows the
    # paper's Table 5 formula literally: the lerp output IS the block size
    # (an integer between 1 and BLOCK_UNIT_CAP), no extra scaling by BUNIT.
    mb_ub = max(1, min(BLOCK_UNIT_CAP, m // (BUNIT * p)))
    mb = max(1, int(round(lerp(alpha, 1, mb_ub))))

    nb_ub = max(1, min(BLOCK_UNIT_CAP, (n * p) // (BUNIT * nproc)))
    nb = max(1, int(round(lerp(gamma, 1, nb_ub))))

    return mb, nb, p, q, npernode, nproc, nthreads


def run_driver(m, n, mb, nb, p, q, nproc, nthreads):
    """Launch the driver, return best PASSED wall time or None on failure."""
    # Run dir on the shared filesystem (visible to all ranks), not /tmp.
    rundir = os.path.join(os.getcwd(), "mlkaps_runs",
                          f"run_{os.getpid()}_{abs(hash((m,n,mb,nb,p,q)))%100000}")
    os.makedirs(rundir, exist_ok=True)
    try:
        with open(os.path.join(rundir, "QR.in"), "w") as f:
            f.write(f"{NITER}\n")
            for _ in range(NITER):
                f.write(
                    f"QR{m:6d}{n:6d}{mb:6d}{nb:6d}{p:6d}{q:6d}"
                    f"{1.0:20.13E}\n"
                )
        cmd = (
            ["srun", "--ntasks", str(nproc), "--cpus-per-task", str(nthreads)]
            + SRUN_FLAGS + [DRIVER, rundir + "/"]
        )
        env = dict(os.environ, OMP_NUM_THREADS=str(nthreads))
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=300, env=env)
        except subprocess.TimeoutExpired:
            return None
        qr_out = os.path.join(rundir, "QR.out")
        if not os.path.exists(qr_out):
            sys.stderr.write("No QR.out\nSTDERR:\n" + proc.stderr + "\n")
            return None
        best = float("inf")
        with open(qr_out) as f:
            for line in f:
                w = line.split()
                if len(w) >= 11 and w[0] == "WALL" and w[9] == "PASSED":
                    best = min(best, float(w[7]))
        return best if math.isfinite(best) else None
    finally:
        shutil.rmtree(rundir, ignore_errors=True)


def main():
    try:
        m = int(float(sys.argv[1]))
        n = int(float(sys.argv[2]))
        alpha = float(sys.argv[3])
        gamma = float(sys.argv[4])
        beta = float(sys.argv[5])
        p_frac = float(sys.argv[6])
    except (IndexError, ValueError) as e:
        sys.stderr.write(f"Bad arguments: {e}\n")
        print(PENALTY)
        return

    mb, nb, p, q, npernode, nproc, nthreads = reformulate(
        m, n, alpha, beta, gamma, p_frac
    )

    sys.stderr.write(f"m={m} n={n} -> mb={mb} nb={nb} p={p} q={q} "
                     f"npernode={npernode} nproc={nproc} nthreads={nthreads}\n")

    # Feasibility is guaranteed by construction
    if nproc < p or p < 1 or q < 1:
        print(PENALTY)
        return

    t = run_driver(m, n, mb, nb, p, q, nproc, nthreads)
    print(t if t is not None else PENALTY)


if __name__ == "__main__":
    main()