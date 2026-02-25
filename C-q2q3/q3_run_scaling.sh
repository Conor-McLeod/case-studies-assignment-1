#!/bin/bash
#SBATCH --job-name=q3-scaling
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=q3_scaling_output.txt
#SBATCH --error=q3_scaling_error.txt

# load intel stack
module load tbb compiler-rt mkl mpi

# Build
make clean && make

# Output file
OUTFILE="q3_scaling_results.csv"
# CSV Column headers. > creates the file if it doesn't exist, overwrites if it does exist.
echo "m,n,time" >"$OUTFILE"

# Warmup run (avoids cold-start penalty on first mpirun)
echo "Warmup run..."
mpirun ./q2_tsqr 100 3 > /dev/null 2>&1

# Test matrix dimensions
M_VALUES="100 1000 10000 100000 1000000"
N_VALUES="3 10 50 100"

for m in $M_VALUES; do
  for n in $N_VALUES; do
    # Skip if n > m/4 (need at least n rows per rank)
    m_local=$((m / 4))
    # Safety check before proceeding. Makes sure blocks are still tall skinny.
    if [ "$n" -gt "$m_local" ]; then
      echo "Skipping m=$m, n=$n (n > m_local=$m_local)"
      continue
    fi

    echo "Running m=$m, n=$n ..."
    output=$(mpirun ./q2_tsqr "$m" "$n" 2>&1)

    # Extract the TIMING line and append to CSV
    timing_line=$(echo "$output" | grep "^TIMING,")
    if [ -n "$timing_line" ]; then
      # TIMING,m,n,time -> m,n,time
      echo "$timing_line" | sed 's/^TIMING,//' >>"$OUTFILE"
    fi
  done
done

echo "Scaling results written to $OUTFILE"
