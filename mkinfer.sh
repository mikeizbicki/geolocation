#!/bin/bash

#for dir in s2 gps_kappa=14b gps_kappa=14c; do
for dir in gps_prekappa0; do
    mkdir -p log/$dir
for f in $(ls output/tfrecord_$dir); do
    time python src/image2/infer.py --initial_weights=output/tfrecord_${dir}/$f --outputdir=log/$dir/$f --batchsize=256
    python src/tensorboard2csv.py log/$dir/$f/train/* log/$dir/$f/results.csv
done
dofirst=True
for f in log/$dir/*/results.csv; do
    if [ $dofirst = True ]; then
        echo "name,$(head -n1 $f)" > log/$dir/$dir.csv
        echo > log/$dir/results.csv.unsorted
        dofirst=False
    fi
    echo "'$f',$(tail -n1 $f)" >> log/$dir/results.csv.unsorted
done
sort -n log/$dir/results.csv.unsorted >> log/$dir/$dir.csv
done
