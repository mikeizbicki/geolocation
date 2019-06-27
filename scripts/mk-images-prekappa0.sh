#!/bin/bash

#img='--img=data/im2gps/104123223_7410c654ba_19_19355699@N00.jpg'
#img='--img=data/im2gps/LosAngeles_00017_271405228_1258990cf2_100_27274415@N00.jpg'
#loc='--s2file=s2/class_cells-8192 --lon_km=36000 --force_center 7 0 --plot_res=hi'
##for prekappa0 in 4 6 8 10 12 14 16 18 20; do
    #prekappa0=10
    #initial_weights="--initial_weights=output/tfrecord_gps_prekappa0/--input_format=tfrecord,--outputs,gps,--gmm_components,8192,--gmm_distribution=fvm2,--s2warmstart_mu,--lores_gmm_prekappa0=$prekappa0/"
    #python src/image2/map.py $initial_weights $loc $opt $img --outputname=$prekappa0 --lores_gmm_prekappa0=8
##done


prekappa0=10
for i in 4 5 6 7 8 9 10 11 12 13 14 15 16 17; do
initial_weights="--initial_weights=output/tfrecord_gps_kappa=14c/--input_format=tfrecord,--outputs,gps,--learningrate=1e-4,--lr_boundaries,1e5,5e5,--lores_gmm_prekappa0=14,--s2warmstart_mu,--gmm_components,$(bc <<< 2^$i)/"
loc='--lon_km=2000 --plot_res=good --lores_gmm_prekappa0=12'
img='--img=data/im2gps/england_00041_251424233_428738c274_111_11114422@N00.jpg'
    python src/image2/map.py $initial_weights $loc $opt $img --outputname=med_12_$i
done
