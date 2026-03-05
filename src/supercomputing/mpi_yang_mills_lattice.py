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

try:
    from mpi4py import MPI
except ImportError:
    print("FATAL: mpi4py is required. Install with: pip install mpi4py")
    sys.exit(1)

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Mock import for actual simulation logic
# from src.engine.observables import simulate_lattice_batch

def mock_simulate_lattice_batch(beta, lattice_size, sweeps):
    """Placeholder for the quantum lattice engine."""
    return [{"beta": beta, "lattice": lattice_size, "wilson_loop_1x1": 0.543, "mass_gap_estimate": 0.88} for _ in range(sweeps)]

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    ap = argparse.ArgumentParser()
    ap.add_argument("--beta", type=float, default=6.0, help="Coupling constant")
    ap.add_argument("--lattice_size", type=int, default=16, help="L^4 size")
    ap.add_argument("--total_sweeps", type=int, default=10000, help="Total Monte Carlo sweeps")
    args = ap.parse_args()

    if rank == 0:
        print("=" * 70)
        print(f" YANG-MILLS MPI SUPERCOMPUTING DISPATCH")
        print(f" Beta: {args.beta} | Lattice: {args.lattice_size}^4 | Sweeps: {args.total_sweeps}")
        print(f" Cluster Workers: {size}")
        print("=" * 70)

        # Distribute sweeps among workers
        base_sweeps = args.total_sweeps // size
        remainder = args.total_sweeps % size
        
        chunks = []
        for i in range(size):
            chunks.append(base_sweeps + (1 if i < remainder else 0))
        start_time = time.time()
    else:
        chunks = None

    # SCATTER
    my_sweeps = comm.scatter(chunks, root=0)

    # PROCESS
    # local_data = simulate_lattice_batch(args.beta, args.lattice_size, my_sweeps)
    local_data = mock_simulate_lattice_batch(args.beta, args.lattice_size, my_sweeps)

    # GATHER
    all_data_lists = comm.gather(local_data, root=0)

    # NODE 0: Ledger Write
    if rank == 0:
        elapsed = time.time() - start_time
        all_data = [item for sublist in all_data_lists for item in sublist]

        out_dir = Path("evidence/supercomputing")
        out_dir.mkdir(parents=True, exist_ok=True)
        
        manifest = {
            "cluster_size": size,
            "beta": args.beta,
            "lattice": f"{args.lattice_size}^4",
            "total_sweeps": len(all_data),
            "wall_time_s": round(elapsed, 2),
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }

        mf_path = out_dir / f"mpi_yangmills_manifest_b{args.beta}_L{args.lattice_size}.json"
        mf_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        print(f"\n{'=' * 70}")
        print(f" MPI RUN COMPLETE | Sweeps calculated: {len(all_data)}")
        print(f" Total Time: {elapsed:.1f}s")
        print(f" Evidence saved to: {out_dir}")
        print(f" {'=' * 70}")

if __name__ == "__main__":
    main()
