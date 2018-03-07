#!/bin/bash

if [ -z $SLURM_SUBMIT_DIR ]; then
    vars=$(tr '/ ' ',' <<< "$@")
    echo $vars
    #sbatch -p gpu --gres=gpu:1 --mem=10g --time=100:00:00 -e "output/${vars}.stderr" -o "output/${vars}.stdout" ./src/train.py $@
    sbatch -J $vars -p gpu --gres=gpu:1 --mem=10g --time=100:00:00 -e "output/${vars}.stderr" -o "output/${vars}.stdout" submit-gpu.sh $@

else
    echo SLURM_JOB_ID=$SLURM_JOB_ID
    module load cuda/9.1
    module load cuDNN/6.0
    . /rhome/mizbicki/tf-gpu-1.4/bin/activate
    ./src/train.py $@
fi

