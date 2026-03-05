#!/bin/bash
#PBS -N Gahenax_YangMills
#PBS -o logs/ym_mpi.out
#PBS -e logs/ym_mpi.err
#PBS -l nodes=4:ppn=32
#PBS -l walltime=72:00:00
#PBS -q batch
#PBS -l pmem=2gb

cd $PBS_O_WORKDIR
module purge
module load python/3.10 openmpi/4.1.4
source venv/bin/activate
export PYTHONPATH=$(pwd)

echo "Starting Gahenax-Yang-Mills on PBS Cluster."
mpiexec python src/supercomputing/mpi_yang_mills_lattice.py --beta 6.0 --lattice_size 16 --total_sweeps 100000
