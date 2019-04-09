#!/bin/bash

function params_gps {
    #echo "--initial_weights=output/tfrecord_gps_kappa=14/--input_format=tfrecord,--outputs,gps,--learningrate=1e-4,--lr_boundaries,1e5,5e5,--lores_gmm_prekappa0=14,--gmm_components,$(bc <<< 2^$1) --s2file=s2/class_cells-$(bc <<< 2^$1) --batchsize=8196"
    echo "--initial_weights=output/tfrecord_gps_kappa=14c/--input_format=tfrecord,--outputs,gps,--learningrate=1e-4,--lr_boundaries,1e5,5e5,--lores_gmm_prekappa0=14,--s2warmstart_mu,--gmm_components,$(bc <<< 2^$1) --batchsize=8196 --s2file=s2/class_cells-$(bc <<< 2^$1)"
}

function params_gps_gpu {
    echo "--initial_weights=output/image_final_bn/--initial_weights=output,image_final.old,1e-4-32768-2e4,--train_only_last_until_step,0,--lr_boundaries,1e5,2e5,--learningrate=1e-5/ --s2file=s2/class_cells-$(bc <<< 2^$1)"
}

function params_s2 {
    echo "--initial_weights=output/tfrecord_s2/--input_format=tfrecord,--outputs,s2,--learningrate=1e-4,--lr_boundaries,1e5,5e5,--s2size=$1 --s2file=s2/class_cells-$(bc <<< 2^$1)"
}

#for f in Switzerland_00086_1367058455_d58f03d50f_1136_9407621@N05.jpg \
    #London_00044_152057337_45584126c5_48_90672364@N00.jpg \
    #London_00124_307812265_70c28bf023_112_30571787@N00.jpg \
    #Chicago_00080_496345089_f5d47dea91_196_69212252@N00.jpg \
    #Colombia_00009_487789777_0003ed5872_212_16836474@N00.jpg \
    #Colorado_00057_1199140107_b35790beec_1001_59341685@N00.jpg \
    #Cuba_00025_699768122_15733c7a1a_1425_8078991@N04.jpg \
    #Switzerland_00014_170747608_cf4538d97b_50_97458173@N00.jpg \
    #Sydney_00073_1226915900_eea86783cd_1128_65768710@N00.jpg \
    #Utah_00026_474986001_181d71a40d_218_53364390@N00.jpg \
    #venice_00002_69431815_c01190b915_35_39303693@N00.jpg \
    #Vermont_00016_746636213_f837f3bf39_1209_19212858@N00.jpg \
    #wales_00004_130253713_3ded2d4456_56_66516604@N00.jpg \
    #LosAngeles_00017_271405228_1258990cf2_100_27274415@N00.jpg \
    #LosAngeles_00023_349121428_aaa440bcc6_132_97332629@N00.jpg \
    #SanFrancisco_00096_442372330_8e060f4138_176_7571568@N05.jpg \
    #California_00197_718693402_4ba930995d_1150_7980350@N08.jpg \
    #Paris_00025_129504349_cdd20e3144_1_42485416@N00.jpg; do
#
##f=LosAngeles_00017_271405228_1258990cf2_100_27274415@N00.jpg
##f=California_00197_718693402_4ba930995d_1150_7980350@N08.jpg
##f=LosAngeles_00023_349121428_aaa440bcc6_132_97332629@N00.jpg
f=SanFrancisco_00096_442372330_8e060f4138_176_7571568@N05.jpg
#f=97344248_30a4521091_32_77325609@N00.jpg
#f=Connecticut_00004_364162616_48c230f9e2_116_15376845@N00.jpg
f=Mexico2_00027_373475154_2cddb4fa40_137_23812473@N00.jpg
#f=Switzerland_00008_103429200_c4904154d8_38_73293249@N00.jpg
#f=england_00041_251424233_428738c274_111_11114422@N00.jpg
#f=UnitedKingdom_00019_964966881_426cf82f57_1071_98545448@N00.jpg
#f=Paris_00025_129504349_cdd20e3144_1_42485416@N00.jpg
#f=California_00197_718693402_4ba930995d_1150_7980350@N08.jpg
#f=Norway_00048_1457219700_4744851a83_1196_99437479@N00.jpg
#f=Italy_00134_844603727_96b5c1d9bc_1229_76306992@N00.jpg
#f=Massachusetts_00011_379112224_fbcf942ebd_169_57253263@N00.jpg
#f=Switzerland_00008_103429200_c4904154d8_38_73293249@N00.jpg
#for f in $f; do
#SanFrancisco_00096_442372330_8e060f4138_176_7571568@N05.jpg
#f=Sydney_00073_1226915900_eea86783cd_1128_65768710@N00.jpg
#UnitedKingdom_00019_964966881_426cf82f57_1071_98545448@N00b.jpg
for f in Mexico2_00027_373475154_2cddb4fa40_137_23812473@N00.jpg ;  do
#for f in $f; do
f=data/im2gps/$f
#f=data/images_presentation/tweet_fire1_img.jpg

#f=data/images_presentation/tweet_fire1_img.jpg

#for f in $(ls -r data/im2gps.maps/*.jpg); do
#for level in 6 7 8 9 10 11 12 13 14 15 16 17; do
#for level in 15 16 17; do
for level in 15;  do
#for method in gps; do
for method in gps_gpu; do
    #opts='--outputdir=data/im2gps.maps'
    opts='--outputdir=img/maps --plot_gps' # --s2file=s2/class_cells-32768' #--plot_sample_points
    #loc='--lon_km=36000 --force_center 7 0 --plot_res=hi --lores_gmm_prekappa0=8'
    #time python src/image2/map.py $(params_${method} $level) $opts $loc --img=$f --outputname=big_${method}_${level}

    #loc='--lon_km=2000 --plot_res=good --lores_gmm_prekappa0=11'
    loc='--lon_km=3500 --plot_res=good --lores_gmm_prekappa0=9'
    time python src/image2/map.py $(params_${method} $level) $opts $loc --img=$f --outputname=med_${method}_${level}

    loc='--lon_km=2000 --plot_res=good --lores_gmm_prekappa0=10'
    time python src/image2/map.py $(params_${method} $level) $opts $loc --img=$f --outputname=med2_${method}_${level}

    #loc='--lon_km=200 --plot_res=med --lores_gmm_prekappa0=14'
    #time python src/image2/map.py $(params_${method} $level) $opts $loc --img=$f --outputname=small_${method}_${level}

    #loc='--lon_km=40 --plot_res=med --lores_gmm_prekappa0=17'
    #time python src/image2/map.py $(params_${method} $level) $opts $loc --img=$f --outputname=tiny_${method}_${level}

    #loc='--lon_km=20 --plot_res=med --s2warmstart_mu --s2warmstart_kappa --s2warmstart_kappa_s=0.01'
    #time python src/image2/map.py $(params_${method} $level) $loc --img=$f --outputname=tiny2_${method}_${level}
done
done
done

#for f in $(ls -r data/im2gps/*.jpg); do
    #opts='--s2file=s2/class_cells-8192'
    #loc='--lon_km=36000 --force_center 7 0 --plot_res=hi --lores_gmm_prekappa0=8'
    #time python src/image2/map.py $initial_weights $opts $loc --img=$f --outputname=big

    #opts='--s2file=s2/class_cells-32768'
    #loc='--lon_km=2000 --plot_res=good --lores_gmm_prekappa0=12'
    #time python src/image2/map.py $initial_weights $opts $loc --img=$f --outputname=med

    #opts='--s2file=s2/class_cells-131072'
    #loc='--lon_km=200 --plot_res=med2'
    #time python src/image2/map.py $initial_weights $opts $loc --img=$f --outputname=small

    #opts='--s2file=s2/class_cells-131072 --s2warmstart_mu --s2warmstart_kappa --s2warmstart_kappa_s=0.5'
    #loc='--lon_km=200 --plot_res=med2'
    #time python src/image2/map.py $initial_weights $opts $loc --img=$f --outputname=zoom2
#done
