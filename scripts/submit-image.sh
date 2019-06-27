#./submit.sh gpu image --model=WideResNet50 --batchsize=64 --learningrate=1e-5 --pretrain --train_only_last_until_step=1e4
#./submit.sh gpu image --model=WideResNet50 --batchsize=64 --learningrate=1e-3 --pretrain --train_only_last_until_step=1e4
#./submit.sh gpu image --model=WideResNet50 --batchsize=64 --learningrate=1e-2 --pretrain --train_only_last_until_step=1e4

#./submit.sh gpu image --model=Inception4 --inputs middles_last --batchsize=64 --learningrate=1e-4 --pretrain --train_only_last_until_step=1e2
#./submit.sh gpu image --model=Inception4 --inputs middles_last --batchsize=64 --learningrate=1e-5 --pretrain --train_only_last_until_step=1e2
#
#./submit.sh gpu image --model=WideResNet50 --inputs middles_last --batchsize=64 --learningrate=1e-4 --pretrain --train_only_last_until_step=1e2
#./submit.sh gpu image --model=WideResNet50 --inputs middles_last --batchsize=64 --learningrate=1e-5 --pretrain --train_only_last_until_step=1e2
#
#./submit.sh gpu image --model=VGG16 --inputs middles_last --batchsize=64 --learningrate=1e-4 --pretrain --train_only_last_until_step=1e2
#./submit.sh gpu image --model=VGG16 --inputs middles_last --batchsize=64 --learningrate=1e-5 --pretrain --train_only_last_until_step=1e2

#./submit.sh gpu image --model=Inception4 --inputs model --batchsize=64 --learningrate=1e-3 --pretrain --train_only_last_until_step=1e4
#./submit.sh gpu image --model=Inception4 --inputs model --batchsize=64 --learningrate=1e-4 --pretrain --train_only_last_until_step=1e4
#./submit.sh gpu image --model=Inception4 --inputs model --batchsize=64 --learningrate=1e-5 --pretrain --train_only_last_until_step=1e4

#./submit.sh gpu image --model=Inception4 --batchsize=64 --learningrate=1e-3
#./submit.sh gpu image --model=Inception4 --batchsize=64 --learningrate=1e-4

#./submit.sh cpu image --inputs --batchsize=64 --learningrate=1e-1
#./submit.sh cpu image --inputs --batchsize=64 --learningrate=1e-2
#./submit.sh cpu image --inputs --batchsize=64 --learningrate=1e-3
#./submit.sh cpu image --inputs --batchsize=64 --learningrate=1e-4
#./submit.sh cpu image --inputs --batchsize=64 --learningrate=1e-5

#./submit.sh gpu image3 --model=WideResNet50 --outputs gps2 country --gmm_type=simple --batchsize=64 --learningrate=1e-5 --trainable_mu --trainable_kappa
#./submit.sh gpu image3 --model=WideResNet50 --outputs gps2 country --gmm_type=simple --batchsize=64 --learningrate=1e-5 --pretrain --train_only_last_until_step=1e3
#./submit.sh gpu image3 --model=WideResNet50 --outputs gps2 country --gmm_type=simple --batchsize=64 --learningrate=5e-6 --pretrain --train_only_last_until_step=1e3
#./submit.sh gpu image3 --model=WideResNet50 --outputs gps2 country --gmm_type=simple --batchsize=64 --learningrate=1e-5 --pretrain --train_only_last_until_step=1e3 --trainable_mu --trainable_kappa
#./submit.sh gpu image3 --model=WideResNet50 --outputs gps country --gmm_type=simple --batchsize=64 --learningrate=1e-5 --pretrain --train_only_last_until_step=1e3 --trainable_mu --trainable_kappa
#
#./submit.sh gpu image3 --model=InceptionResNet2 --outputs gps country --gmm_type=simple --batchsize=64 --learningrate=1e-5 --pretrain --train_only_last_until_step=1e3
#./submit.sh gpu image3 --model=InceptionResNet2 --outputs gps country --gmm_type=simple --batchsize=64 --learningrate=1e-4 --pretrain --train_only_last_until_step=1e3

#./submit.sh gpu image --initial_weights=/rhome/mizbicki/bigdata/geolocating/output/best-early/image
#./submit.sh gpu image --initial_weights=/rhome/mizbicki/bigdata/geolocating/output/best-early/image --trainable_mu
#./submit.sh gpu image --initial_weights=/rhome/mizbicki/bigdata/geolocating/output/best-early/image --trainable_kappa
#./submit.sh gpu image --initial_weights=/rhome/mizbicki/bigdata/geolocating/output/best-early/image --trainable_mu --trainable_kappa
#./submit.sh gpu image --initial_weights=/rhome/mizbicki/bigdata/geolocating/output/best-early/image --learningrate=1e-4 --trainable_mu --trainable_kappa
#./submit.sh gpu image --initial_weights=/rhome/mizbicki/bigdata/geolocating/output/best-early/image --learningrate=1e-4

#./submit.sh gpu image --initial_weights=/rhome/mizbicki/bigdata/geolocating/output/best-early/image2 --trainable_weights=False --trainable_mu=True --train_only_last_until_step=1e50
#./submit.sh gpu image --initial_weights=/rhome/mizbicki/bigdata/geolocating/output/best-early/image2 --trainable_weights=False --trainable_mu=True --trainable_kappa=True --train_only_last_until_step=1e50
#./submit.sh gpu image --initial_weights=/rhome/mizbicki/bigdata/geolocating/output/best-early/image2 --trainable_weights=False --trainable_mu=True --train_only_last_until_step=1e50 --learningrate=1e-6

#./submit.sh gpu image --initial_weights=/rhome/mizbicki/bigdata/geolocating/output/best-early/image-k10 --trainable_weights=True --trainable_mu=False --train_only_last_until_step=0
#./submit.sh gpu image --initial_weights=/rhome/mizbicki/bigdata/geolocating/output/best-early/image2 --lores_gmm_prekappa0=10.0 --trainable_weights=True --trainable_mu=True --trainable_kappa=False --train_only_last_until_step=0 --gmm_minimizedist
#./submit.sh gpu image --initial_weights=/rhome/mizbicki/bigdata/geolocating/output/best-early/image2 --lores_gmm_prekappa0=10.0 --trainable_weights=True --trainable_mu=True --trainable_kappa=True --train_only_last_until_step=0 --gmm_minimizedist

#./submit.sh gpu image_splitter --initial_weights=output/best-early/image3 --outputs country gps2b --train_only_last_until=500000 --lores_gmm_prekappa0=10.0 --hires_gmm_prekappa0=12.0
#./submit.sh gpu image_splitter --initial_weights=output/best-early/image3 --outputs country gps2b --train_only_last_until=500000 --lores_gmm_prekappa0=11.0 --hires_gmm_prekappa0=13.0
#./submit.sh gpu image_splitter --initial_weights=output/best-early/image3 --outputs country gps2b --train_only_last_until=500000 --lores_gmm_prekappa0=12.0 --hires_gmm_prekappa0=14.0

#./submit.sh gpu image_splitter --initial_weights=output/best-early/image3 --outputs country gps2b --train_only_last_until=500000 --lores_gmm_prekappa0=8.0 --hires_gmm_prekappa0=9.0 #--trainable_mu=True --trainable_kappa=True
#./submit.sh gpu image_splitter --initial_weights=output/best-early/image3 --outputs country gps2b --train_only_last_until=500000 --lores_gmm_prekappa0=6.0 --hires_gmm_prekappa0=7.0 #--trainable_mu=True --trainable_kappa=True
#./submit.sh gpu image_splitter --initial_weights=output/best-early/image3 --outputs country gps2b --train_only_last_until=500000 --lores_gmm_prekappa0=5.0 --hires_gmm_prekappa0=6.0 #--trainable_mu=True --trainable_kappa=True

#./submit.sh gpu image_splitter --initial_weights=output/best-early/image3 --outputs country gps2b --train_only_last_until=500000 --lores_gmm_prekappa0=-4.0 --hires_gmm_prekappa0=-3.0 #--trainable_mu=True --trainable_kappa=True
#./submit.sh gpu image_splitter --initial_weights=output/best-early/image3 --outputs country gps2b --train_only_last_until=500000 --lores_gmm_prekappa0=-6.0 --hires_gmm_prekappa0=-5.0 #--trainable_mu=True --trainable_kappa=True

#./submit.sh gpu image_splitter --initial_weights=output/best-early/image3 --outputs country gps2b --train_only_last_until=500000 --lores_gmm_prekappa0=2.0 --hires_gmm_prekappa0=3.0
#./submit.sh gpu image_splitter --initial_weights=output/best-early/image3 --outputs country gps2b --train_only_last_until=500000 --lores_gmm_prekappa0=2.0 --hires_gmm_prekappa0=3.0 --gmm_xentropy

#./submit.sh gpu image_splitter --initial_weights=output/best-early/image3 --outputs country gps2b --train_only_last_until=500000 --lores_gmm_prekappa0=7.0 --hires_gmm_prekappa0=8.0
#./submit.sh gpu image_splitter --initial_weights=output/best-early/image3 --outputs country gps2b --train_only_last_until=500000 --lores_gmm_prekappa0=7.0 --hires_gmm_prekappa0=8.0 --gmm_xentropy

#./submit.sh gpu image_splitter --initial_weights=output/best-early/image3 --outputs country gps2b --train_only_last_until=500000 --lores_gmm_prekappa0=10.0 --hires_gmm_prekappa0=12.0
#./submit.sh gpu image_splitter --initial_weights=output/best-early/image3 --outputs country gps2b --train_only_last_until=500000 --lores_gmm_prekappa0=10.0 --hires_gmm_prekappa0=12.0 --gmm_xentropy

#./submit.sh gpu image_splitter --initial_weights=output/best-early/image3 --outputs country gps2b --train_only_last_until=5000000 --gmm_xentropy --gmm_no_logloss

#./submit.sh gpu image_splitter --train_only_last_until=5000000 --initial_weights=output/image_splitter/--initial_weights=output,best-early,image3,--outputs,country,gps2b,--train_only_last_until=500000,--lores_gmm_prekappa0=10.0,--hires_gmm_prekappa0=12.0
#./submit.sh gpu image_splitter --train_only_last_until=5000000 --initial_weights=output/image_splitter/--initial_weights=output,best-early,image3,--outputs,country,gps2b,--train_only_last_until=500000,--lores_gmm_prekappa0=10.0,--hires_gmm_prekappa0=12.0,--gmm_xentropy
#./submit.sh gpu image_splitter --train_only_last_until=5000000 --initial_weights=output/image_splitter/--initial_weights=output,best-early,image3,--outputs,country,gps2b,--train_only_last_until=500000,--lores_gmm_prekappa0=2.0,--hires_gmm_prekappa0=3.0
#./submit.sh gpu image_splitter --train_only_last_until=5000000 --initial_weights=output/image_splitter/--initial_weights=output,best-early,image3,--outputs,country,gps2b,--train_only_last_until=500000,--lores_gmm_prekappa0=2.0,--hires_gmm_prekappa0=3.0,--gmm_xentropy
#./submit.sh gpu image_splitter --train_only_last_until=5000000 --initial_weights=output/image_splitter/--initial_weights=output,best-early,image3,--outputs,country,gps2b,--train_only_last_until=500000,--lores_gmm_prekappa0=7.0,--hires_gmm_prekappa0=8.0
#./submit.sh gpu image_splitter --train_only_last_until=5000000 --initial_weights=output/image_splitter/--initial_weights=output,best-early,image3,--outputs,country,gps2b,--train_only_last_until=500000,--lores_gmm_prekappa0=7.0,--hires_gmm_prekappa0=8.0,--gmm_xentropy

#./submit.sh gpu image_new --initial_weights=output/best-new/image-complex2 --trainable_mu=False
#./submit.sh gpu image_final --initial_weights=output/best-new/image-complex2 --trainable_mu=False --learningrate=5e-5 --reset_global_step --output gps --gmm_components $(bc <<< '2^13') --s2warmstart_mu --lores_gmm_prekappa0=14 --train_only_last_until_step 2e4 --lr_boundaries 2e4 2e5
#./submit.sh gpu image_final --initial_weights=output/best-new/image-complex2 --trainable_mu=False --learningrate=5e-5 --reset_global_step --output gps --gmm_components $(bc <<< '2^17') --s2warmstart_mu --lores_gmm_prekappa0=14 --train_only_last_until_step 2e4 --lr_boundaries 2e4 2e5
#./submit.sh gpu image_final --initial_weights=output/best-new/image-complex2 --trainable_mu=False --learningrate=1e-4 --reset_global_step --output gps --gmm_components $(bc <<< '2^17') --s2warmstart_mu --lores_gmm_prekappa0=14 --train_only_last_until_step 2e4 --lr_boundaries 2e4 2e5
#./submit.sh gpu image_final --initial_weights=output/best-new/image-complex2 --trainable_mu=False --learningrate=5e-5 --reset_global_step --output gps --gmm_components $(bc <<< '2^15') --s2warmstart_mu --lores_gmm_prekappa0=14 --train_only_last_until_step 2e4 --lr_boundaries 2e4 2e5
#./submit.sh gpu image_final --initial_weights=output/best-new/image-complex2 --trainable_mu=False --learningrate=1e-4 --reset_global_step --output gps --gmm_components $(bc <<< '2^15') --s2warmstart_mu --lores_gmm_prekappa0=14 --train_only_last_until_step 2e4 --lr_boundaries 2e4 2e5

#./submit.sh gpu image_final_bn --initial_weights=output/best-new/image-complex2 --trainable_mu=False --learningrate=1e-4 --reset_global_step --output gps --gmm_components $(bc <<< '2^13') --s2warmstart_mu --lores_gmm_prekappa0=14 --train_only_last_until_step 1e6 --lr_boundaries 1e5 2e5

#./submit.sh gpu image_final_bn --initial_weights=output/image_final.old/1e-4-8192-2e4 --train_only_last_until_step 0 --lr_boundaries 1e5 2e5 --learningrate=1e-4
#./submit.sh gpu image_final_bn --initial_weights=output/image_final.old/1e-4-8192-2e4 --train_only_last_until_step 0 --lr_boundaries 1e5 2e5 --learningrate=1e-5
#./submit.sh gpu image_final_bn --initial_weights=output/image_final.old/1e-4-8192-2e4 --train_only_last_until_step 0 --lr_boundaries 1e5 2e5 --learningrate=1e-6

#./submit.sh gpu image_final_bn --initial_weights=output/image_final.old/5e-5-32768-2e4  --lr_boundaries 1e5 2e5
#./submit.sh gpu image_final_bn --initial_weights=output/image_final.old/1e-4-32768-2e4  --train_only_last_until_step 0 --lr_boundaries 1e5 2e5 --learningrate=1e-3
#./submit.sh gpu image_final_bn --initial_weights=output/image_final.old/1e-4-32768-2e4  --train_only_last_until_step 0 --lr_boundaries 1e5 2e5 --learningrate=1e-4
#./submit.sh gpu image_final_bn --initial_weights=output/image_final.old/1e-4-32768-2e4  --train_only_last_until_step 0 --lr_boundaries 1e5 2e5 --learningrate=1e-5
#./submit.sh gpu image_final_bn --initial_weights=output/image_final.old/1e-4-32768-2e4  --train_only_last_until_step 0 --lr_boundaries 1e5 2e5 --learningrate=1e-6

#./submit.sh gpu image_final_bn --initial_weights=output/image_final.old/5e-5-131072-2e4  --lr_boundaries 1e5 2e5
#./submit.sh gpu image_final_bn --initial_weights=output/image_final.old/1e-4-131072-2e4 --train_only_last_until_step 0 --lr_boundaries 1e5 2e5 --learningrate=1e-4 --batchsize=32
#./submit.sh gpu image_final_bn --initial_weights=output/image_final.old/1e-4-131072-2e4 --train_only_last_until_step 0 --lr_boundaries 1e5 2e5 --learningrate=1e-5 --batchsize=32
#./submit.sh gpu image_final_bn --initial_weights=output/image_final.old/1e-4-131072-2e4 --train_only_last_until_step 0 --lr_boundaries 1e5 2e5 --learningrate=1e-6 --batchsize=32

#./submit.sh gpu image_final_bn3 --initial_weights=output/image_final_bn/--initial_weights=output,image_final.old,1e-4-8192-2e4,--train_only_last_until_step,0,--lr_boundaries,1e5,2e5,--learningrate=1e-5  --trainable_mu=True --mu_max_meters=1
./submit.sh gpu image_final_bn3 --initial_weights=output/image_final_bn/--initial_weights=output,image_final.old,1e-4-32768-2e4,--train_only_last_until_step,0,--lr_boundaries,1e5,2e5,--learningrate=1e-5/  --trainable_mu=True --mu_max_meters=1
#./submit.sh gpu image_final_bn3 --initial_weights=output/image_final_bn/--initial_weights=output,image_final.old,1e-4-131072-2e4,--train_only_last_until_step,0,--lr_boundaries,1e5,2e5,--learningrate=1e-5,--batchsize=32 --trainable_mu=True --mu_max_meters=1
#./submit.sh gpu image_final_bn3 --initial_weights=output/image_final_bn/--initial_weights=output,image_final.old,1e-4-8192-2e4,--train_only_last_until_step,0,--lr_boundaries,1e5,2e5,--learningrate=1e-5  --trainable_mu=True --mu_max_meters=0.01
#./submit.sh gpu image_final_bn3 --initial_weights=output/image_final_bn/--initial_weights=output,image_final.old,1e-4-32768-2e4,--train_only_last_until_step,0,--lr_boundaries,1e5,2e5,--learningrate=1e-5/  --trainable_mu=True --mu_max_meters=0.01
#./submit.sh gpu image_final_bn3 --initial_weights=output/image_final_bn/--initial_weights=output,image_final.old,1e-4-131072-2e4,--train_only_last_until_step,0,--lr_boundaries,1e5,2e5,--learningrate=1e-5,--batchsize=32 --trainable_mu=True --mu_max_meters=0.01

#./submit.sh gpu image_final_bn2 --initial_weights=output/image_final_bn/--initial_weights\=output\,image_final.old\,1e-4-8192-2e4\,--train_only_last_until_step\,0\,--lr_boundaries\,1e5\,2e5\,--learningrate\=1e-5/ --trainable_mu=True --mu_max_meters=0.001
#./submit.sh gpu image_final_bn2 --initial_weights=output/image_final_bn/--initial_weights\=output\,image_final.old\,1e-4-32768-2e4\,--train_only_last_until_step\,0\,--lr_boundaries\,1e5\,2e5\,--learningrate\=1e-5/ --trainable_mu=True --mu_max_meters=0.001
#./submit.sh gpu image_final_bn2 --initial_weights=output/image_final_bn/--initial_weights\=output\,image_final.old\,1e-4-131072-2e4\,--train_only_last_until_step\,0\,--lr_boundaries\,1e5\,2e5\,--learningrate\=1e-5,--batchsize\=32/ --trainable_mu=True --mu_max_meters=0.001
#./submit.sh gpu image_final_bn2 --initial_weights=output/image_final_bn/--initial_weights\=output\,image_final.old\,1e-4-8192-2e4\,--train_only_last_until_step\,0\,--lr_boundaries\,1e5\,2e5\,--learningrate\=1e-5/ --trainable_mu=True --mu_max_meters=0.01
#./submit.sh gpu image_final_bn2 --initial_weights=output/image_final_bn/--initial_weights\=output\,image_final.old\,1e-4-32768-2e4\,--train_only_last_until_step\,0\,--lr_boundaries\,1e5\,2e5\,--learningrate\=1e-5/ --trainable_mu=True --mu_max_meters=0.01
#./submit.sh gpu image_final_bn2 --initial_weights=output/image_final_bn/--initial_weights\=output\,image_final.old\,1e-4-131072-2e4\,--train_only_last_until_step\,0\,--lr_boundaries\,1e5\,2e5\,--learningrate\=1e-5,--batchsize\=32/ --trainable_mu=True --mu_max_meters=0.01

#./submit.sh gpu image_final_bn --initial_weights=output/image_final_bn/--initial_weights\=output\,image_final.old\,1e-4-131072-2e4\,--train_only_last_until_step\,0\,--lr_boundaries\,1e5\,2e5\,--learningrate\=1e-5,--batchsize\=32/ --lores_gmm_prekappa0=14 --trainable_mu=True --mu_max_meters=0.0001
#./submit.sh gpu image_final_bn --initial_weights=output/image_final_bn/--initial_weights\=output\,image_final.old\,1e-4-131072-2e4\,--train_only_last_until_step\,0\,--lr_boundaries\,1e5\,2e5\,--learningrate\=1e-5,--batchsize\=32/ --lores_gmm_prekappa0=15 --lrb=2
#./submit.sh gpu image_final_bn --initial_weights=output/image_final_bn/--initial_weights\=output\,image_final.old\,1e-4-131072-2e4\,--train_only_last_until_step\,0\,--lr_boundaries\,1e5\,2e5\,--learningrate\=1e-5,--batchsize\=32/ --lores_gmm_prekappa0=16 --lrb=2
#./submit.sh gpu image_final_bn --initial_weights=output/image_final_bn/--initial_weights\=output\,image_final.old\,1e-4-131072-2e4\,--train_only_last_until_step\,0\,--lr_boundaries\,1e5\,2e5\,--learningrate\=1e-5,--batchsize\=32/ --lores_gmm_prekappa0=12 --lrb=2

#./submit.sh gpu image_final_bn --initial_weights=output/image_final_bn/--initial_weights\=output\,image_final.old\,1e-4-32768-2e4\,--train_only_last_until_step\,0\,--lr_boundaries\,1e5\,2e5\,--learningrate\=1e-5/ --lores_gmm_prekappa0=12 --lrb=2
#./submit.sh gpu image_final_bn --initial_weights=output/image_final_bn/--initial_weights\=output\,image_final.old\,1e-4-32768-2e4\,--train_only_last_until_step\,0\,--lr_boundaries\,1e5\,2e5\,--learningrate\=1e-5/ --lores_gmm_prekappa0=15 --lrb=2
