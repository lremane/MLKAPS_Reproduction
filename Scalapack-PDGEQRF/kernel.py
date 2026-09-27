#!/usr/bin/env python3
# MLKAPS wrapper for ScaLAPACK PDGEQRF on one JURECA node.
# MLKAPS only handles unconstrained inputs in [0,1], so we map them to a valid
# configuration with the lerp trick from Table 5 of the MLKAPS paper.

import sys
import os
import subprocess
import shutil
import math

NODES = 1
CORES = 128
BUNIT = 8
BLOCK_UNIT_CAP = 16
NITER = 3
PENALTY = 1e12

DRIVER = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "scalapack-driver", "bin", "jsc", "pdqrdriver",
)
SRUN_FLAGS = ["--exact", "--overlap", "--mpi=pspmix"]


def lerp(t, lb, ub):
    if ub < lb:
        ub = lb
    t = max(0.0, min(1.0, t))
    return lb + t * (ub - lb)


def reformulate(m, n, alpha, beta, gamma, p_frac):
    nproc = int(round(lerp(beta, 1, CORES)))
    nproc = max(1, min(CORES, nproc))

    divs = [d for d in range(1, nproc + 1) if nproc % d == 0]
    p = divs[int(round(lerp(p_frac, 0, len(divs) - 1)))]
    q = nproc // p

    mb_ub = max(1, min(BLOCK_UNIT_CAP, m // (BUNIT * p)))
    mb = max(1, int(round(lerp(alpha, 1, mb_ub))))

    nb_ub = max(1, min(BLOCK_UNIT_CAP, (n * p) // (BUNIT * nproc)))
    nb = max(1, int(round(lerp(gamma, 1, nb_ub))))

    return mb, nb, p, q, nproc


def run_driver(m, n, mb, nb, p, q, nproc):
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

        # TODO: support mor than 1 thread per rank
        cmd = (
                ["srun", "--ntasks", str(nproc), "--cpus-per-task", "1"]
                + SRUN_FLAGS + [DRIVER, rundir + "/"]
        )
        env = dict(os.environ, OMP_NUM_THREADS="1")
        
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
        m, n = int(float(sys.argv[1])), int(float(sys.argv[2]))
        alpha, gamma, beta, p_frac = map(float, sys.argv[3:7])
    except (IndexError, ValueError) as e:
        sys.stderr.write(f"Bad arguments: {e}\n")
        print(PENALTY)
        return

    mb, nb, p, q, nproc = reformulate(m, n, alpha, beta, gamma, p_frac)
    print(f"m={m} n={n} mb={mb} nb={nb} p={p} q={q} nproc={nproc}", file=sys.stderr)

    if nproc < p or p < 1 or q < 1:
        print(PENALTY)
        return

    t = run_driver(m, n, mb, nb, p, q, nproc)
    print(t if t is not None else PENALTY)


if __name__ == "__main__":
    main()