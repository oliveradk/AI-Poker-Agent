#!/bin/bash
#SBATCH --job-name=poker_mcts        # Job name
#SBATCH --output=output_log/job_%j.out      # Output file (%j expands to jobID)
#SBATCH --error=output_log/job_%j.err       # Error file (%j expands to jobID)
#SBATCH --ntasks=1               # Number of tasks
#SBATCH --cpus-per-task=1        # Number of CPU cores per task
#SBATCH --mem=1G                 # Memory required (1 GB in this example)
#SBATCH --time=24:00:00          # Time limit (HH:MM:SS)

module load conda/latest
conda activate poker_mcts

python train.py

