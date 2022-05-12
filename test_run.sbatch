#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00
#SBATCH --mem=8GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=E100_Adam5e_3
#SBATCH --mail-type=END
#SBATCH --mail-user=zm2114@nyu.edu
#SBATCH --output=slurm_%j.out

module purge
module load cuda/11.3.1 
source activate /scratch/fda239/penv2

cd /scratch/fda239/Kaggle
python main.py 