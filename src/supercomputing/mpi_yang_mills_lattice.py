"""
MPI Multi-Node Dispatch for Yang-Mills SU(3) Lattice Gauge Simulation.

Distributes Monte Carlo sweeps and Wilson Loop calculations across 
multiple physical cluster nodes to scale beyond single-machine limits.
"""
import sys
import argparse
import time
import json
from pathlib import Path
from datetime import datetime

import numpy as np

try:
    from mpi4py import MPI
except ImportError:
    print("FATAL: mpi4py is required. Install with: pip install mpi4py")
    sys.exit(1)

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def simulate_lattice_batch(
    betas: list,
    N: int,
    dim: int,
    n_sweeps: int,
    n_configs: int,
    seed_offset: int = 0,
) -> list:
    """
    Run thermalized Yang-Mills simulations for a list of beta values.

    For each beta: creates a LatticeGauge, thermalizes with mixed sweeps
    (overrelaxation + Metropolis), then collects n_configs measurements of
    the average plaquette and action, returning mean + bootstrap error.

    Parameters
    ----------
    betas       : list of coupling constants to simulate
    N           : lattice size per dimension
    dim         : number of spacetime dimensions
    n_sweeps    : thermalization sweeps (mixed)
    n_configs   : measurement configurations after thermalization
    seed_offset : base seed (rank * large_prime added for independence)

    Returns
    -------
    list of dicts with keys: beta, plaquette_mean, plaquette_err,
                             action_mean, action_err, n_configs
    """
    from lattice_gauge import LatticeGauge
    from statistics import bootstrap_error

    results = []
    for i, beta in enumerate(betas):
        seed = seed_offset + i * 1000003  # large prime gap for independence
        lat = LatticeGauge(N, dim, beta=beta, seed=seed)

        # Thermalize with mixed sweeps
        for _ in range(n_sweeps):
            lat.mixed_sweep(n_over=4)

        # Collect measurements
        plaqs = []
        actions = []
        for _ in range(n_configs):
            lat.mixed_sweep(n_over=4)
            plaqs.append(lat.average_plaquette())
            actions.append(lat.plaquette_action())

        p_mean, p_err = bootstrap_error(plaqs)
        a_mean, a_err = bootstrap_error(actions)

        results.append({
            "beta": beta,
            "plaquette_mean": round(p_mean, 6),
            "plaquette_err": round(p_err, 6),
            "action_mean": round(a_mean, 4),
            "action_err": round(a_err, 4),
            "n_configs": n_configs,
        })
    return results


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    ap = argparse.ArgumentParser()
    ap.add_argument("--beta_min", type=float, default=1.0, help="Minimum beta")
    ap.add_argument("--beta_max", type=float, default=4.0, help="Maximum beta")
    ap.add_argument("--n_betas", type=int, default=16, help="Number of beta values")
    ap.add_argument("--lattice_size", type=int, default=4, help="L^dim size")
    ap.add_argument("--dim", type=int, default=4, help="Spacetime dimensions")
    ap.add_argument("--n_sweeps", type=int, default=50, help="Thermalization sweeps")
    ap.add_argument("--n_configs", type=int, default=50, help="Measurement configs")
    args = ap.parse_args()

    if rank == 0:
        print("=" * 70)
        print(f" YANG-MILLS MPI SUPERCOMPUTING DISPATCH")
        print(f" Beta range: [{args.beta_min}, {args.beta_max}] ({args.n_betas} values)")
        print(f" Lattice: {args.lattice_size}^{args.dim} | Cluster Workers: {size}")
        print("=" * 70)

        all_betas = list(np.linspace(args.beta_min, args.beta_max, args.n_betas))
        # Distribute betas as evenly as possible among workers
        chunks = [[] for _ in range(size)]
        for j, b in enumerate(all_betas):
            chunks[j % size].append(b)
        start_time = time.time()
    else:
        chunks = None
        start_time = None

    # SCATTER: each worker gets its list of betas
    my_betas = comm.scatter(chunks, root=0)

    # PROCESS: real simulation
    local_results = simulate_lattice_batch(
        betas=my_betas,
        N=args.lattice_size,
        dim=args.dim,
        n_sweeps=args.n_sweeps,
        n_configs=args.n_configs,
        seed_offset=rank * 999983,
    )

    # GATHER
    all_results_lists = comm.gather(local_results, root=0)

    # NODE 0: Aggregate and save
    if rank == 0:
        elapsed = time.time() - start_time
        all_results = sorted(
            [item for sublist in all_results_lists for item in sublist],
            key=lambda x: x["beta"],
        )

        out_dir = Path("evidence/supercomputing")
        out_dir.mkdir(parents=True, exist_ok=True)

        manifest = {
            "cluster_size": size,
            "beta_range": [args.beta_min, args.beta_max],
            "n_betas": args.n_betas,
            "lattice": f"{args.lattice_size}^{args.dim}",
            "n_sweeps": args.n_sweeps,
            "n_configs": args.n_configs,
            "total_simulations": len(all_results),
            "wall_time_s": round(elapsed, 2),
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "results": all_results,
        }

        mf_path = (out_dir /
                   f"mpi_yangmills_b{args.beta_min}-{args.beta_max}"
                   f"_L{args.lattice_size}.json")
        mf_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        print(f"\n{'=' * 70}")
        print(f" MPI RUN COMPLETE | Simulations: {len(all_results)}")
        print(f" Total Time: {elapsed:.1f}s")
        print(f" Results saved to: {mf_path}")
        print(f"{'=' * 70}")

if __name__ == "__main__":
    main()
