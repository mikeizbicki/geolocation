#!/bin/bash

params="${@:3}"
output=output/$2
mkdir -p $output

if [ -z $SLURM_SUBMIT_DIR ]; then

    vars=$(tr '/ ' ',' <<< "$params")
    echo $vars

    mkdir -p $output/$vars
    opts_debug="-J $vars -e $output/$vars/stderr -o $output/$vars/stdout"
    if [[ $1 =~ "gpu" ]]; then
        numgpus='--gres=gpu:1'
        if [[ $1 = 'gpu2' ]]; then numgpus='--gres=gpu:2'; fi
        if [[ $1 = 'gpu3' ]]; then numgpus='--gres=gpu:3'; fi
        if [[ $1 = 'gpu4' ]]; then numgpus='--gres=gpu:4'; fi
        if [[ $1 = 'gpu6' ]]; then numgpus='--gres=gpu:6 --cpus-per-task=10 --ntasks=1'; fi
        opts_gpu="-p gpu $numgpus "
    else
        if [[ $1 = 'cpu' ]]; then opts_gpu=''; fi
        if [[ $1 = 'cpu2' ]]; then opts_gpu='--cpus-per-task=2 --ntasks=1'; fi
        if [[ $1 = 'cpu4' ]]; then opts_gpu='--cpus-per-task=4 --ntasks=1'; fi
        if [[ $1 = 'cpu8' ]]; then opts_gpu='--cpus-per-task=8 --ntasks=1'; fi
    fi
    opts_slurm="--mem=50g --time=28-0:00:00"
    opts_log="--log_dir=$output --log_name=${vars}"
    sbatch $opts_debug $opts_gpu $opts_slurm scripts/submit.sh $@ $opts_log

else
    echo SLURM_JOB_ID=$SLURM_JOB_ID
    if [[ $1 =~ 'gpu' ]]; then
        module load cuda/9.1
        module load cuDNN/6.0
        . /rhome/mizbicki/tf-gpu-1.4/bin/activate
    else
        . /rhome/mizbicki/tf-cpu-1.4/bin/activate
    fi
    python -u ./src/model/train.py $params
    #python -u ./src/image2/train.py $params
fi

