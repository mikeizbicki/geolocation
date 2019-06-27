#task='tfrecord_gps_17'
#task='tfrecord_s2'
#task='tfrecord_gps_prekappa0'
task='tfrecord_gps_kappa=14c_mu'
#task='tfrecord_gps_mix'
#task='tfrecord_gps_mu'

if [[ $task = tfrecord_gps_mu ]]; then
    common='--input_format=tfrecord --initial_weights=output/best-new/tfrecord-gps14'
    #./submit.sh cpu2 $task $common --trainable_weights=True --trainable_kappa=False --trainable_mu=False
    #./submit.sh cpu2 $task $common --trainable_weights=True --trainable_kappa=True --trainable_mu=False

    #./submit.sh cpu2 $task $common --trainable_weights=True --trainable_kappa=True --trainable_mu=True --mu_max_meters=0.1
    #./submit.sh cpu2 $task $common --trainable_weights=True --trainable_kappa=True --trainable_mu=True --mu_max_meters=0.01
    #./submit.sh cpu2 $task $common --trainable_weights=True --trainable_kappa=True --trainable_mu=True --mu_max_meters=0.001
    ./submit.sh cpu2 $task $common --trainable_weights=True --trainable_kappa=True --trainable_mu=True --mu_max_meters=1

    #./submit.sh cpu2 $task $common --trainable_weights=False --trainable_kappa=True --trainable_mu=True --mu_max_meters=0.1
    #./submit.sh cpu2 $task $common --trainable_weights=False --trainable_kappa=True --trainable_mu=True --mu_max_meters=0.01
    #./submit.sh cpu2 $task $common --trainable_weights=False --trainable_kappa=True --trainable_mu=True --mu_max_meters=0.001
    ./submit.sh cpu2 $task $common --trainable_weights=False --trainable_kappa=True --trainable_mu=True --mu_max_meters=1

    #./submit.sh cpu2 $task $common --trainable_weights=True --trainable_kappa=True --trainable_mu=False --s2warmstart_mu --s2warmstart_kappa --s2warmstart_kappa_s=0.5
    #./submit.sh cpu2 $task $common --trainable_weights=True --trainable_kappa=True --trainable_mu=False --s2warmstart_mu --s2warmstart_kappa --s2warmstart_kappa_s=0.9
    #./submit.sh cpu2 $task $common --trainable_weights=True --trainable_kappa=True --trainable_mu=False --s2warmstart_mu --s2warmstart_kappa --s2warmstart_kappa_s=0.99

    #./submit.sh cpu2 $task $common --trainable_weights=True --trainable_kappa=True --trainable_mu=True --s2warmstart_mu --s2warmstart_kappa --s2warmstart_kappa_s=0.5 --mu_max_meters=0.001
    #./submit.sh cpu2 $task $common --trainable_weights=True --trainable_kappa=True --trainable_mu=True --s2warmstart_mu --s2warmstart_kappa --s2warmstart_kappa_s=0.9 --mu_max_meters=0.001
    #./submit.sh cpu2 $task $common --trainable_weights=True --trainable_kappa=True --trainable_mu=True --s2warmstart_mu --s2warmstart_kappa --s2warmstart_kappa_s=0.99 --mu_max_meters=0.001

fi

if [[ $task = tfrecord_gps_prekappa0_mu ]]; then
    ./submit.sh cpu2 $task --input_format=tfrecord --initial_weights=output/best-new/tfrecord-gps14 --lores_gmm_prekappa0=14

    common='--input_format=tfrecord --initial_weights=output/best-new/tfrecord-gps14 --trainable_weights=True --trainable_mu=True --trainable_kappa=True'
    ./submit.sh cpu2 $task $common --lores_gmm_prekappa0=5  --gmm_xentropy
    ./submit.sh cpu2 $task $common --lores_gmm_prekappa0=10 --gmm_xentropy
    ./submit.sh cpu2 $task $common --lores_gmm_prekappa0=15 --gmm_xentropy
    ./submit.sh cpu2 $task $common --lores_gmm_prekappa0=20 --gmm_xentropy
    ./submit.sh cpu2 $task $common --lores_gmm_prekappa0=25 --gmm_xentropy

    common='--input_format=tfrecord --initial_weights=output/best-new/tfrecord-gps14 --trainable_weights=False --trainable_mu=True --trainable_kappa=False'
    ./submit.sh cpu2 $task $common --lores_gmm_prekappa0=5  --gmm_xentropy
    ./submit.sh cpu2 $task $common --lores_gmm_prekappa0=10 --gmm_xentropy
    ./submit.sh cpu2 $task $common --lores_gmm_prekappa0=15 --gmm_xentropy
    ./submit.sh cpu2 $task $common --lores_gmm_prekappa0=20 --gmm_xentropy
    ./submit.sh cpu2 $task $common --lores_gmm_prekappa0=25 --gmm_xentropy

    common='--input_format=tfrecord --initial_weights=output/best-new/tfrecord-gps14 --trainable_weights=True --trainable_mu=False --trainable_kappa=False'
    ./submit.sh cpu2 $task $common --lores_gmm_prekappa0=5  --gmm_xentropy
    ./submit.sh cpu2 $task $common --lores_gmm_prekappa0=10 --gmm_xentropy
    ./submit.sh cpu2 $task $common --lores_gmm_prekappa0=15 --gmm_xentropy
    ./submit.sh cpu2 $task $common --lores_gmm_prekappa0=20 --gmm_xentropy
    ./submit.sh cpu2 $task $common --lores_gmm_prekappa0=25 --gmm_xentropy
fi

if [[ $task = tfrecord_gps_gradient ]]; then
    common='--input_format=tfrecord --initial_weights=output/best-new/tfrecord-gps14' # --gmm_xentropy'
    ./submit.sh cpu2 $task $common --learningrate=1e-5 --trainable_weights=False --trainable_mu=True --trainable_kappa=False --gmm_gradient_method=stop
    ./submit.sh cpu2 $task $common --learningrate=1e-5 --trainable_weights=False --trainable_mu=True --trainable_kappa=False --gmm_gradient_method=main
    ./submit.sh cpu2 $task $common --learningrate=1e-5 --trainable_weights=False --trainable_mu=True --trainable_kappa=False --gmm_gradient_method=all

    ./submit.sh cpu2 $task $common --learningrate=1e-5 --trainable_weights=False --trainable_mu=False --trainable_kappa=True --gmm_gradient_method=stop
    ./submit.sh cpu2 $task $common --learningrate=1e-5 --trainable_weights=False --trainable_mu=False --trainable_kappa=True --gmm_gradient_method=main
    ./submit.sh cpu2 $task $common --learningrate=1e-5 --trainable_weights=False --trainable_mu=False --trainable_kappa=True --gmm_gradient_method=all

    ./submit.sh cpu2 $task $common --learningrate=1e-5 --trainable_weights=True --trainable_mu=True --trainable_kappa=True --gmm_gradient_method=stop
    ./submit.sh cpu2 $task $common --learningrate=1e-5 --trainable_weights=True --trainable_mu=True --trainable_kappa=True --gmm_gradient_method=main
    ./submit.sh cpu2 $task $common --learningrate=1e-5 --trainable_weights=True --trainable_mu=True --trainable_kappa=True --gmm_gradient_method=all
fi

if [[ $task = tfrecord_gps_mix ]]; then
    common='--input_format=tfrecord --initial_weights=output/best-new/tfrecord-gps14'
    ./submit.sh cpu2 $task $common --learningrate=1e-5 --trainable_weights=False --trainable_mu=True --trainable_kappa=False
    ./submit.sh cpu2 $task $common --learningrate=1e-6 --trainable_weights=False --trainable_mu=True --trainable_kappa=False
    ./submit.sh cpu2 $task $common --learningrate=1e-7 --trainable_weights=False --trainable_mu=True --trainable_kappa=False
    ./submit.sh cpu2 $task $common --learningrate=1e-8 --trainable_weights=False --trainable_mu=True --trainable_kappa=False

    ./submit.sh cpu2 $task $common --learningrate=1e-5 --trainable_weights=False --trainable_mu=True --trainable_kappa=True
    ./submit.sh cpu2 $task $common --learningrate=1e-6 --trainable_weights=False --trainable_mu=True --trainable_kappa=True
    ./submit.sh cpu2 $task $common --learningrate=1e-7 --trainable_weights=False --trainable_mu=True --trainable_kappa=True
    ./submit.sh cpu2 $task $common --learningrate=1e-8 --trainable_weights=False --trainable_mu=True --trainable_kappa=True

    ./submit.sh cpu2 $task $common --learningrate=1e-5 --trainable_weights=False --trainable_mu=False --trainable_kappa=True
    ./submit.sh cpu2 $task $common --learningrate=1e-6 --trainable_weights=False --trainable_mu=False --trainable_kappa=True
    ./submit.sh cpu2 $task $common --learningrate=1e-7 --trainable_weights=False --trainable_mu=False --trainable_kappa=True
    ./submit.sh cpu2 $task $common --learningrate=1e-8 --trainable_weights=False --trainable_mu=False --trainable_kappa=True

    ./submit.sh cpu2 $task $common --trainable_weights=True --trainable_mu=True --trainable_kappa=True
    ./submit.sh cpu2 $task $common --trainable_weights=True --trainable_mu=True --trainable_kappa=False
    ./submit.sh cpu2 $task $common --trainable_weights=True --trainable_mu=False --trainable_kappa=True

    ./submit.sh cpu2 $task $common --gmm_xentropy --trainable_weights=True --trainable_mu=True --trainable_kappa=True
    ./submit.sh cpu2 $task $common --gmm_xentropy --trainable_weights=True --trainable_mu=True --trainable_kappa=False
    ./submit.sh cpu2 $task $common --gmm_xentropy --trainable_weights=True --trainable_mu=False --trainable_kappa=True
    ./submit.sh cpu2 $task $common --gmm_xentropy --trainable_weights=True --trainable_mu=False --trainable_kappa=False
fi

# run all s2 learning
if [[ $task = tfrecord_s2 ]]; then
    common='--input_format=tfrecord --outputs s2 --learningrate=1e-4 --lr_boundaries 1e5 5e5'
    ./submit.sh cpu $task $common --s2size=4
    ./submit.sh cpu $task $common --s2size=5
    ./submit.sh cpu $task $common --s2size=6
    ./submit.sh cpu $task $common --s2size=7
    ./submit.sh cpu2 $task $common --s2size=8
    ./submit.sh cpu2 $task $common --s2size=9
    ./submit.sh cpu4 $task $common --s2size=10
    ./submit.sh cpu4 $task $common --s2size=11
    ./submit.sh cpu8 $task $common --s2size=12
    ./submit.sh cpu8 $task $common --s2size=13
    ./submit.sh gpu $task $common --s2size=14
    ./submit.sh gpu $task $common --s2size=15
    ./submit.sh gpu $task $common --s2size=16
    ./submit.sh gpu $task $common --s2size=17
fi

if [[ $task = tfrecord_gps_kappa=14c_mu ]]; then
    #./submit.sh cpu $task --initial_weights=output/tfrecord_gps_kappa\=14c/--input_format\=tfrecord\,--outputs\,gps\,--learningrate\=1e-4\,--lr_boundaries\,1e5\,5e5\,--lores_gmm_prekappa0\=14\,--s2warmstart_mu\,--gmm_components\,$(bc <<< 2^4) --trainable_mu=True --trainable_kappa=True
    #./submit.sh cpu $task --initial_weights=output/tfrecord_gps_kappa\=14c/--input_format\=tfrecord\,--outputs\,gps\,--learningrate\=1e-4\,--lr_boundaries\,1e5\,5e5\,--lores_gmm_prekappa0\=14\,--s2warmstart_mu\,--gmm_components\,$(bc <<< 2^5) --trainable_mu=True --trainable_kappa=True
    #./submit.sh cpu $task --initial_weights=output/tfrecord_gps_kappa\=14c/--input_format\=tfrecord\,--outputs\,gps\,--learningrate\=1e-4\,--lr_boundaries\,1e5\,5e5\,--lores_gmm_prekappa0\=14\,--s2warmstart_mu\,--gmm_components\,$(bc <<< 2^6) --trainable_mu=True --trainable_kappa=True
    #./submit.sh cpu $task --initial_weights=output/tfrecord_gps_kappa\=14c/--input_format\=tfrecord\,--outputs\,gps\,--learningrate\=1e-4\,--lr_boundaries\,1e5\,5e5\,--lores_gmm_prekappa0\=14\,--s2warmstart_mu\,--gmm_components\,$(bc <<< 2^7) --trainable_mu=True --trainable_kappa=True
    #./submit.sh cpu2 $task --initial_weights=output/output/tfrecord_gps_kappa\=14c/--input_format\=tfrecord\,--outputs\,gps\,--learningrate\=1e-4\,--lr_boundaries\,1e5\,5e5\,--lores_gmm_prekappa0\=14\,--s2warmstart_mu\,--gmm_components\,$(bc <<< 2^8) --trainable_mu=True --trainable_kappa=True
    #./submit.sh cpu2 $task --initial_weights=output/tfrecord_gps_kappa\=14c/--input_format\=tfrecord\,--outputs\,gps\,--learningrate\=1e-4\,--lr_boundaries\,1e5\,5e5\,--lores_gmm_prekappa0\=14\,--s2warmstart_mu\,--gmm_components\,$(bc <<< 2^9) --trainable_mu=True --trainable_kappa=True
    #./submit.sh cpu4 $task --initial_weights=output/tfrecord_gps_kappa\=14c/--input_format\=tfrecord\,--outputs\,gps\,--learningrate\=1e-4\,--lr_boundaries\,1e5\,5e5\,--lores_gmm_prekappa0\=14\,--s2warmstart_mu\,--gmm_components\,$(bc <<< 2^10) --trainable_mu=True --trainable_kappa=True
    #./submit.sh cpu4 $task --initial_weights=output/tfrecord_gps_kappa\=14c/--input_format\=tfrecord\,--outputs\,gps\,--learningrate\=1e-4\,--lr_boundaries\,1e5\,5e5\,--lores_gmm_prekappa0\=14\,--s2warmstart_mu\,--gmm_components\,$(bc <<< 2^11) --trainable_mu=True --trainable_kappa=True
    #./submit.sh cpu8 $task --initial_weights=output/tfrecord_gps_kappa\=14c/--input_format\=tfrecord\,--outputs\,gps\,--learningrate\=1e-4\,--lr_boundaries\,1e5\,5e5\,--lores_gmm_prekappa0\=14\,--s2warmstart_mu\,--gmm_components\,$(bc <<< 2^12) --trainable_mu=True --trainable_kappa=True
    #./submit.sh cpu8 $task --initial_weights=output/tfrecord_gps_kappa\=14c/--input_format\=tfrecord\,--outputs\,gps\,--learningrate\=1e-4\,--lr_boundaries\,1e5\,5e5\,--lores_gmm_prekappa0\=14\,--s2warmstart_mu\,--gmm_components\,$(bc <<< 2^13) --trainable_mu=True --trainable_kappa=True
    #./submit.sh gpu $task --initial_weights=output/tfrecord_gps_kappa\=14c/--input_format\=tfrecord\,--outputs\,gps\,--learningrate\=1e-4\,--lr_boundaries\,1e5\,5e5\,--lores_gmm_prekappa0\=14\,--s2warmstart_mu\,--gmm_components\,$(bc <<< 2^14) --trainable_mu=True --trainable_kappa=True
    #./submit.sh gpu $task --initial_weights=output/tfrecord_gps_kappa\=14c/--input_format\=tfrecord\,--outputs\,gps\,--learningrate\=1e-4\,--lr_boundaries\,1e5\,5e5\,--lores_gmm_prekappa0\=14\,--s2warmstart_mu\,--gmm_components\,$(bc <<< 2^15) --trainable_mu=True --trainable_kappa=True
    ./submit.sh gpu $task --initial_weights=output/tfrecord_gps_kappa\=14c/--input_format\=tfrecord\,--outputs\,gps\,--learningrate\=1e-4\,--lr_boundaries\,1e5\,5e5\,--lores_gmm_prekappa0\=14\,--s2warmstart_mu\,--gmm_components\,$(bc <<< 2^16) --trainable_mu=True --trainable_kappa=True
    ./submit.sh gpu $task --initial_weights=output/tfrecord_gps_kappa\=14c/--input_format\=tfrecord\,--outputs\,gps\,--learningrate\=1e-4\,--lr_boundaries\,1e5\,5e5\,--lores_gmm_prekappa0\=14\,--s2warmstart_mu\,--gmm_components\,$(bc <<< 2^17) --trainable_mu=True --trainable_kappa=True
fi

if [[ $task = tfrecord_gps_kappa=14c ]]; then
    common="--input_format=tfrecord --outputs gps --learningrate=1e-4 --lr_boundaries 1e5 5e5 --lores_gmm_prekappa0=14 --s2warmstart_mu"
    ./submit.sh cpu $task $common --gmm_components $(bc <<<'2^4')
    ./submit.sh cpu $task $common --gmm_components $(bc <<<'2^5')
    ./submit.sh cpu $task $common --gmm_components $(bc <<<'2^6')
    ./submit.sh cpu $task $common --gmm_components $(bc <<<'2^7')
    ./submit.sh cpu2 $task $common --gmm_components $(bc <<<'2^8')
    ./submit.sh cpu2 $task $common --gmm_components $(bc <<<'2^9')
    ./submit.sh cpu4 $task $common --gmm_components $(bc <<<'2^10')
    ./submit.sh cpu4 $task $common --gmm_components $(bc <<<'2^11')
    ./submit.sh cpu8 $task $common --gmm_components $(bc <<<'2^12')
    ./submit.sh cpu8 $task $common --gmm_components $(bc <<<'2^13')
    ./submit.sh gpu $task $common --gmm_components $(bc <<<'2^14')
    ./submit.sh gpu $task $common --gmm_components $(bc <<<'2^15')
    ./submit.sh gpu $task $common --gmm_components $(bc <<<'2^16')
    ./submit.sh gpu $task $common --gmm_components $(bc <<<'2^17')
fi

if [[ $task = tfrecord_gps_prekappa0_efam ]]; then
    common="--input_format=tfrecord --outputs gps --gmm_components $(bc <<< '2^13') --gmm_distribution=fvm2 --gmm_gradient_method=efam --trainable_kappa=True"
    ./submit.sh cpu2 $task $common --s2warmstart_mu --lores_gmm_prekappa0=4
    ./submit.sh cpu2 $task $common --s2warmstart_mu --lores_gmm_prekappa0=6
    ./submit.sh cpu2 $task $common --s2warmstart_mu --lores_gmm_prekappa0=8
    ./submit.sh cpu2 $task $common --s2warmstart_mu --lores_gmm_prekappa0=10
    ./submit.sh cpu2 $task $common --s2warmstart_mu --lores_gmm_prekappa0=12
    ./submit.sh cpu2 $task $common --s2warmstart_mu --lores_gmm_prekappa0=14
    ./submit.sh cpu2 $task $common --s2warmstart_mu --lores_gmm_prekappa0=16
    ./submit.sh cpu2 $task $common --s2warmstart_mu --lores_gmm_prekappa0=18
    ./submit.sh cpu2 $task $common --s2warmstart_mu --lores_gmm_prekappa0=20

    ./submit.sh cpu2 $task $common --s2warmstart_mu --s2warmstart_kappa --s2warmstart_kappa_s=0.0000001
    ./submit.sh cpu2 $task $common --s2warmstart_mu --s2warmstart_kappa --s2warmstart_kappa_s=0.000001
    ./submit.sh cpu2 $task $common --s2warmstart_mu --s2warmstart_kappa --s2warmstart_kappa_s=0.00001
    ./submit.sh cpu2 $task $common --s2warmstart_mu --s2warmstart_kappa --s2warmstart_kappa_s=0.0001
    ./submit.sh cpu2 $task $common --s2warmstart_mu --s2warmstart_kappa --s2warmstart_kappa_s=0.001
    ./submit.sh cpu2 $task $common --s2warmstart_mu --s2warmstart_kappa --s2warmstart_kappa_s=0.01
    ./submit.sh cpu2 $task $common --s2warmstart_mu --s2warmstart_kappa --s2warmstart_kappa_s=0.1
    ./submit.sh cpu2 $task $common --s2warmstart_mu --s2warmstart_kappa --s2warmstart_kappa_s=0.5
    ./submit.sh cpu2 $task $common --s2warmstart_mu --s2warmstart_kappa --s2warmstart_kappa_s=0.9
    ./submit.sh cpu2 $task $common --s2warmstart_mu --s2warmstart_kappa --s2warmstart_kappa_s=0.99
    ./submit.sh cpu2 $task $common --s2warmstart_mu --s2warmstart_kappa --s2warmstart_kappa_s=0.999
    #./submit.sh cpu2 $task $common --s2warmstart_mu --s2warmstart_kappa --s2warmstart_kappa_s=0.9999
    #./submit.sh cpu2 $task $common --s2warmstart_mu --s2warmstart_kappa --s2warmstart_kappa_s=0.99999
fi


if [[ $task = tfrecord_gps_prekappa0 ]]; then
    common="--input_format=tfrecord --outputs gps --gmm_components $(bc <<< '2^13') --gmm_distribution=fvm2"
    ./submit.sh cpu2 $task $common --s2warmstart_mu --lores_gmm_prekappa0=0
    ./submit.sh cpu2 $task $common --s2warmstart_mu --lores_gmm_prekappa0=2
    #./submit.sh cpu2 $task $common --s2warmstart_mu --lores_gmm_prekappa0=4
    #./submit.sh cpu2 $task $common --s2warmstart_mu --lores_gmm_prekappa0=6
    #./submit.sh cpu2 $task $common --s2warmstart_mu --lores_gmm_prekappa0=8
    #./submit.sh cpu2 $task $common --s2warmstart_mu --lores_gmm_prekappa0=10
    #./submit.sh cpu2 $task $common --s2warmstart_mu --lores_gmm_prekappa0=12
    #./submit.sh cpu2 $task $common --s2warmstart_mu --lores_gmm_prekappa0=14
    #./submit.sh cpu2 $task $common --s2warmstart_mu --lores_gmm_prekappa0=16
    #./submit.sh cpu2 $task $common --s2warmstart_mu --lores_gmm_prekappa0=18
    #./submit.sh cpu2 $task $common --s2warmstart_mu --lores_gmm_prekappa0=20
    ./submit.sh cpu2 $task $common --s2warmstart_mu --lores_gmm_prekappa0=22
    ./submit.sh cpu2 $task $common --s2warmstart_mu --lores_gmm_prekappa0=24
    ./submit.sh cpu2 $task $common --s2warmstart_mu --lores_gmm_prekappa0=26
    ./submit.sh cpu2 $task $common --s2warmstart_mu --lores_gmm_prekappa0=28
    ./submit.sh cpu2 $task $common --s2warmstart_mu --lores_gmm_prekappa0=30
    ./submit.sh cpu2 $task $common --s2warmstart_mu --lores_gmm_prekappa0=-10
    ./submit.sh cpu2 $task $common --s2warmstart_mu --lores_gmm_prekappa0=-8
    ./submit.sh cpu2 $task $common --s2warmstart_mu --lores_gmm_prekappa0=-6
    ./submit.sh cpu2 $task $common --s2warmstart_mu --lores_gmm_prekappa0=-4
    ./submit.sh cpu2 $task $common --s2warmstart_mu --lores_gmm_prekappa0=-2
    ./submit.sh cpu2 $task $common --s2warmstart_mu --lores_gmm_prekappa0=32
    ./submit.sh cpu2 $task $common --s2warmstart_mu --lores_gmm_prekappa0=34
    ./submit.sh cpu2 $task $common --s2warmstart_mu --lores_gmm_prekappa0=36
    ./submit.sh cpu2 $task $common --s2warmstart_mu --lores_gmm_prekappa0=38
    ./submit.sh cpu2 $task $common --s2warmstart_mu --lores_gmm_prekappa0=40

    #./submit.sh cpu2 $task $common --s2warmstart_mu --s2warmstart_kappa --s2warmstart_kappa_s=0.0000001
    #./submit.sh cpu2 $task $common --s2warmstart_mu --s2warmstart_kappa --s2warmstart_kappa_s=0.000001
    #./submit.sh cpu2 $task $common --s2warmstart_mu --s2warmstart_kappa --s2warmstart_kappa_s=0.00001
    #./submit.sh cpu2 $task $common --s2warmstart_mu --s2warmstart_kappa --s2warmstart_kappa_s=0.0001
    #./submit.sh cpu2 $task $common --s2warmstart_mu --s2warmstart_kappa --s2warmstart_kappa_s=0.001
    #./submit.sh cpu2 $task $common --s2warmstart_mu --s2warmstart_kappa --s2warmstart_kappa_s=0.01
    #./submit.sh cpu2 $task $common --s2warmstart_mu --s2warmstart_kappa --s2warmstart_kappa_s=0.1
    #./submit.sh cpu2 $task $common --s2warmstart_mu --s2warmstart_kappa --s2warmstart_kappa_s=0.5
    #./submit.sh cpu2 $task $common --s2warmstart_mu --s2warmstart_kappa --s2warmstart_kappa_s=0.9
    #./submit.sh cpu2 $task $common --s2warmstart_mu --s2warmstart_kappa --s2warmstart_kappa_s=0.99
    #./submit.sh cpu2 $task $common --s2warmstart_mu --s2warmstart_kappa --s2warmstart_kappa_s=0.999
    #./submit.sh cpu2 $task $common --s2warmstart_mu --s2warmstart_kappa --s2warmstart_kappa_s=0.9999
    #./submit.sh cpu2 $task $common --s2warmstart_mu --s2warmstart_kappa --s2warmstart_kappa_s=0.99999
fi

if [[ $task = tfrecord_gps_17 ]]; then
    common="--input_format=tfrecord --outputs gps --gmm_components $(bc <<< '2^17') --gmm_distribution=fvm2"
    #./submit.sh gpu $task $common
    ./submit.sh gpu $task $common --s2warmstart_mu --lores_gmm_prekappa0=6
    ./submit.sh gpu $task $common --s2warmstart_mu --lores_gmm_prekappa0=10
    ./submit.sh gpu $task $common --s2warmstart_mu --lores_gmm_prekappa0=14
    ./submit.sh gpu $task $common --s2warmstart_mu --lores_gmm_prekappa0=18
    #./submit.sh gpu $task $common --s2warmstart_mu --s2warmstart_kappa
    #./submit.sh gpu $task $common --s2warmstart_mu --trainable_mu=True

    common="--input_format=tfrecord --outputs gps --gmm_components $(bc <<< '2^15')"
    #./submit.sh gpu $task $common --s2warmstart_mu --gmm_xentropy
    #./submit.sh gpu $task $common --s2warmstart_mu --trainable_mu=True --learningrate=1e-4 --lores_gmm_prekappa0=20
fi

#task=tfrecord_lr
#
#common='--input_format=tfrecord --outputs gps country --learningrate=1e-5'
#
#for i in $(seq 4 18); do
    #gmm_components=$(bc <<< "2^$i")
    #./submit.sh cpu $task $common --gmm_components=$gmm_components
#done

#for i in $(seq 1 16); do
    #./submit.sh cpu $task $common --gmm_components=8096 --lores_gmm_prekappa0=$i
#done
#
#./submit.sh gpu tfrecord_lr_gpu $common --gmm_components=8096 --lores_gmm_prekappa0=4
#./submit.sh gpu tfrecord_lr_gpu $common --gmm_components=8096 --lores_gmm_prekappa0=8
#./submit.sh gpu tfrecord_lr_gpu $common --gmm_components=8096 --lores_gmm_prekappa0=12

#common='--input_format=tfrecord --outputs s2 --s2file=s2/class_cells-8192 --learningrate=1e-5'
#./submit.sh gpu tfrecord_s2 $common
