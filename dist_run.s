#!/bin/bash
#SBATCH --nodes=2                # node count
#SBATCH --ntasks-per-node=2      # total number of tasks per node
#SBATCH --cpus-per-task=4        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --time=00:01:00
#SBATCH --mem=2GB
#SBATCH --gres=gpu:2
#SBATCH --job-name=torch_farris
#SBATCH --mail-type=END
#SBATCH --mail-user=fda239@nyu.edu
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
srun python princeton.py --epochs=2 --batch-size=2 --lr=.1