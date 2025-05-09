#!/bin/bash
#SBATCH --job-name=poker_mcts        # Job name
#SBATCH --output=output_log/eval_job_%j.out      # Output file (%j expands to jobID)
#SBATCH --error=output_log/eval_job_%j.err       # Error file (%j expands to jobID)
#SBATCH --ntasks=1               # Number of tasks
#SBATCH --cpus-per-task=32        # Number of CPU cores per task
#SBATCH --mem=1G                 # Memory required (1 GB in this example)
#SBATCH --time=12:00:00          # Time limit (HH:MM:SS)

module load conda/latest
conda activate poker_mcts

exp_dir=$1
num_procs=$2
max_rounds=$3

python tournament_eval.py --exp_dir=$exp_dir --processes=$num_procs --max_rounds=$max_rounds 
