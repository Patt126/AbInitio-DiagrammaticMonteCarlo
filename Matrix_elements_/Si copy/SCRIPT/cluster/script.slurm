#!/bin/sh
#SBATCH --job-name=plotbands
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24       # INCREASED: Use more cores (e.g., 12, 24, or 48)
#SBATCH --mem=74G
#SBATCH --time=12:00:00
#SBATCH --qos=skylake_0096
#SBATCH --partition=skylake_0096
#SBATCH --hint=nomultithread     # NEW: Turn off hyperthreading for pure speed
#SBATCH --output=script.out
#SBATCH --error=script.err

# --- Environment Optimization ---

# 1. Map OpenMP threads to the requested CPUs
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

# 2. Pin threads to cores (Crucial for Intel CPUs)
# This prevents the OS from moving threads between cores, preserving L1/L2 cache
export KMP_AFFINITY=granularity=fine,compact,1,0

# 3. Print config for debugging
echo "Running on node: $(hostname)"
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"

# --- Run Script ---
python3 -u main.py
