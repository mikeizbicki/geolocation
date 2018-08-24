#!/bin/bash

outdir=img/tables
mkdir -p outdir
#./src/plots/eventreader.py --outdir=img/tables --models output/best/lang
#./src/plots/eventreader.py --outdir=img/tables --models output/best/* output/fullrun2/* output/ltb_new/--data=data,BillionTwitter,--input,lang,time,bow,--full,--output,pos,country,--learningrate=1.03E-03,--l2=1.66E-06,--l1=5.37E-01,--decay=4.31E+03,--pos_type=aglm_mix,--bow_dense,--summary_newusers

#./src/plots/eventreader.py --burnin=100 --outdir=img/tables --models output/best/cnn-med output/fullrun2/* output/newest_ltb_256/* output/newest_lt/* output/newest_l/*
./src/plots/eventreader.py --burnin=100 --outdir=img/tables --models output/newest_ltb_256/--data=data,BillionTwitter,--input,lang,time,bow,--full,--output,pos,country,--learningrate=1.40E-03,--l2=1.01E-06,--l1=1.15E-01,--decay=2.60E+04,--pos_type=aglm_mix,--no_staging_area,--summary_newusers,--bow_layersize=256 output/newest_lt/--data=data,BillionTwitter,,--input,lang,time,--output,pos,country,--learningrate=1.49E-03,--pos_type=aglm_mix,--no_staging_area,--summary_newusers output/newest_l/--data=data,BillionTwitter,,--input,lang,--output,pos,country,--learningrate=7.79E-04,--pos_type=aglm_mix,--no_staging_area,--summary_newusers output/best/cnn-med output/fullrun2/*
#./src/plots/eventreader.py --outdir=img/tables --models output/new_ltb/*
