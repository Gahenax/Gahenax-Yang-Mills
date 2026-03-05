#!/bin/bash
#SBATCH --job-name=Gahenax_YangMills
#SBATCH --output=logs/ym_mpi_%j.out
#SBATCH --nodes=4                   
#SBATCH --ntasks-per-node=32        
#SBATCH --time=72:00:00             
#SBATCH --partition=compute         

module purge
module load python/3.10 openmpi/4.1.4
source venv/bin/activate
export PYTHONPATH=$(pwd)

echo "Starting Gahenax-Yang-Mills on $SLURM_JOB_NUM_NODES nodes."
mpirun python src/supercomputing/mpi_yang_mills_lattice.py --beta 6.0 --lattice_size 16 --total_sweeps 100000
