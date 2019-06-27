#!/bin/bash

#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=5G
#SBATCH --time=2-00:00:00

module load cuda/9.1
module load cuDNN/6.0
. /rhome/mizbicki/tf-gpu-1.4/bin/activate
$@

