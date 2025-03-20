#!/bin/bash -e

#SBATCH --job-name=icml # create a short name for your job
#SBATCH --output=logs/%A.out # create a output file
#SBATCH --error=logs/%A.err # create a error file
#SBATCH --partition=movianr # choose partition
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=64
#SBATCH --mem=256GB
#SBATCH --exclude=sdc2-hpc-dgx-a100-015
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail          # send email when job fails
#SBATCH --mail-user=v.ngannh9@vinai.io

eval "$(conda shell.bash hook)"
conda deactivate
conda deactivate
conda activate /lustre/scratch/client/movian/research/users/ngannh9/Infinity/env/flashattn
# CUDA_VISIBLE_DEVICES=0,1,2,3 bash distill_controlnet.sh
cd /lustre/scratch/client/movian/research/users/ngannh9/Infinity
export XDG_CACHE_HOME=/lustre/scratch/client/movian/research/users/ngannh9/.cache
# bash train.sh

scripts/trainc.sh