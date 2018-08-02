#!/bin/bash

outdir=img/tables
mkdir -p outdir
#./src/plots/eventreader.py --outdir=img/tables --models output/best/lang
./src/plots/eventreader.py --outdir=img/tables --models output/best/* output/fullrun2/* output/ltb_new/--data=data,BillionTwitter,--input,lang,time,bow,--full,--output,pos,country,--learningrate=1.03E-03,--l2=1.66E-06,--l1=5.37E-01,--decay=4.31E+03,--pos_type=aglm_mix,--bow_dense,--summary_newusers
