#!/bin/bash

sbatch scripts/passthrough-gpu.sh python ./src/model/infer.py --modeldir=models/unicodecnn-large --overwrite --noplots --tweets=$1
