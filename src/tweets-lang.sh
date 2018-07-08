#!/bin/bash

lang=$1
data=data/twitter-early

for file in $(find $data | grep 'gz$'); do
    echo $file
    newdir=$(sed -e "s;$data;${data}.$lang;" <<< $(dirname $file))
    mkdir -p $newdir
    gunzip -c $file | grep -e "\"lang\": \"$lang\"" | gzip > $newdir/$(basename $file)
done

