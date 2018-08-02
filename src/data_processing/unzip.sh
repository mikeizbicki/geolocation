#!/bin/bash

for zipfile in *.zip; do
    dirname=$(cut -d. -f1 <<< $zipfile)
    if [ ! -d $dirname ]; then
        unzip $zipfile
        for txtfile in $dirname/*; do
            gzip $txtfile
        done
        rm $zipfile
    else
        echo skipping ${dirname}
    fi
done
