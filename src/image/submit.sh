#!/bin/bash

#SBATCH --mem-per-cpu=20G
#SBATCH --time=10-00:15:00
#SBATCH --gres=gpu:1
#SBATCH -p gpu
#SBATCH --cpus-per-task=4

if [[ -e $SLUTM_SUBMIT_DIR ]]; then
    cd $SLURM_SUBMIT_DIR
fi

module load cuda/9.1
module load cuDNN/6.0
. /rhome/mizbicki/tf-gpu-1.4/bin/activate

python ./src/image2/train.py $@
