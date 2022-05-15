#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:v100:1
#SBATCH --job-name=E100_Adam5e3
#SBATCH --mail-type=END
#SBATCH --mail-user=zm2114@nyu.edu
#SBATCH --output=slurm_%j.out

export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "WORLD_SIZE="$WORLD_SIZE

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

module purge
module load cuda/11.3.1 
source activate /scratch/fda239/penv2

cd /scratch/fda239/Kaggle
python main_all.py 