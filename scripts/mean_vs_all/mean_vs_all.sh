#!/bin/sh

#SBATCH --account=def-amw8
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rtao3@ualberta.ca
#SBATCH --error=/home/taodav/scratch/log/ble/slurm-%j-%n-%a.err
#SBATCH --output=/home/taodav/scratch/log/ble/slurm-%j-%n-%a.out
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1-00:00
#SBATCH --array=1-4,12

# Load CUDA and CuDNN for tf
module load cuda/11.2 cudnn

# make our log directory if it doesn't exist
#mkdir -p /home/taodav/scratch/log/ble

cd ../../  # Go to main project folder
source venv/bin/activate

TO_RUN=$(sed -n "${SLURM_ARRAY_TASK_ID}p" scripts/mean_vs_all/mean_vs_all_runs.txt)
eval $TO_RUN