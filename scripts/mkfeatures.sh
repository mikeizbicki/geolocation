#!/bin/bash

#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=10-00:00:00

module load cuda/9.1
module load cuDNN/6.0
. /rhome/mizbicki/tf-gpu-1.4/bin/activate

python src/image2/train.py --initial_weights=output/best-new/image-complex2/ --log_dir=log --log_name=mkfeatures --features_file=im2gps.tfrecord2 --batchsize=128 --outputs=gps --jpg_dir='/rhome/mizbicki/bigdata/geolocating/data/im2gps'

