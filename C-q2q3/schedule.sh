#!/bin/bash
#SBATCH --job-name=case-studies-1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:02:00
#SBATCH --output=%x_output.txt
#SBATCH --error=%x_error.txt

# load intel stack
module load tbb compiler-rt mkl mpi

# Run the executable (already compiled)
mpirun ./q2_tsqr
