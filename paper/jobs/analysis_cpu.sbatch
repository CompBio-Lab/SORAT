#!/bin/bash
# CPU job wrapper for the slower analysis scripts. Usage (from paper/):
#   sbatch --account=<cpu-account> jobs/analysis_cpu.sbatch analysis/slice_metrics.py
#SBATCH --job-name=sorat_paper
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.log
cd "$SLURM_SUBMIT_DIR"
./run_analysis.sh "$@"
