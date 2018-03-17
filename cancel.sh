#!/bin/bash

for run in $@; do
    echo $run
    slurmid=$(head -n1 "$run/stdout" | grep SLURM | cut -d = -f 2)
    echo slurmid=$slurmid
    qdel $slurmid
    mkdir -p $(dirname ${run}).old
    mv $run $(dirname ${run}).old/$(basename $run)
done
