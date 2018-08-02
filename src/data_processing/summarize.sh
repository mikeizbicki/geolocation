#!/bin/bash
. ~/tf-cpu-1.4/bin/activate

for dir in $(find ./data/BillionTwitter/g* -type d); do
    pickle="${dir}.pkl"
    if [ ! -f $pickle ]; then
        echo "$dir : summarizing"
        touch $pickle
        ./src/data_processing/summarize.py --outfile=$pickle --files $dir/* #--verbose > ${pickle}.stdout
    else
        echo "$dir : skipping"
    fi
done
