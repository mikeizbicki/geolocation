#!/bin/bash

params="${@:2}"
output=output
mkdir -p $output

if [ -z $SLURM_SUBMIT_DIR ]; then

    vars=$(tr '/ ' ',' <<< "$params")
    echo $vars

    opts_debug="-J $vars -e $output/${vars}.stderr -o $output/${vars}.stdout"
    if [[ $1 = "gpu" ]]; then
        opts_gpu="-p gpu --gres=gpu:1"
    fi
    opts_slurm="--mem=20g --time=100:00:00"
    opts_log="--log_dir=$output --log_name=${vars}"
    sbatch $opts_debug $opts_gpu $opts_slurm submit.sh $@ $opts_log

else
    echo SLURM_JOB_ID=$SLURM_JOB_ID
    if [[ $1 = 'gpu' ]]; then
        module load cuda/9.1
        module load cuDNN/6.0
        . /rhome/mizbicki/tf-gpu-1.4/bin/activate
    else
        . /rhome/mizbicki/tf-cpu-1.4/bin/activate
    fi
    ./src/train.py $params
fi

