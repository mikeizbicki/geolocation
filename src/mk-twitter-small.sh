#/bin/bash

twitter_dir='/rhome/mizbicki/twitter2'
output_dir='./data/twitter'

mkdir -p $output_dir

for i in $twitter_dir/geoT*; do
    if [[ -d $i ]]; then
        echo $(basename $i)
        mkdir -p $output_dir/$(basename $i)
        for j in $i/*; do
            printf "   $(basename $j) ... "
            outfile=$output_dir/$(basename $i)/$(basename $j)
            if [[ ! -f ${outfile}.gz && ! -f $outfile ]]; then
                #head -n 10000 $j > $outfile
                python ./src/json-filter.py < $j > $outfile
                gzip $outfile &
                echo done
            else
                echo skip
            fi
        done
    fi
done
