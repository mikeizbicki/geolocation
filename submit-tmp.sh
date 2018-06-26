
task='dist'

################################################################################
if [[ $task = decompose ]]; then
    common="--data=data/twitter-early --input cnn lang time --full 2048 2048 --output country pos --pos_type=aglm_mix --gmm_type=verysimple --learningrate=5e-4 --decay=1e5 --gmm_prekappa0=10.0 --gmm_components=10000"

    ./submit.sh gpu $task $common --gmm_decompose 16
    ./submit.sh gpu $task $common --gmm_decompose 16 256
    ./submit.sh gpu $task $common --gmm_decompose 16 256 4096
    ./submit.sh gpu $task $common --gmm_decompose 256
    ./submit.sh gpu $task $common --gmm_decompose 32 1024

################################################################################
elif [[ $task = early_mix ]]; then
    common="--data=data/twitter-early --pos_type=aglm_mix --gmm_type=verysimple --learningrate=5e-4 --decay=1e5"

    #./submit.sh gpu $task $common --gmm_prekappa0=10.0 --gmm_components=10000 --initial_weights=output/early_mix/--gmm_prekappa0=10.0/model.ckpt-370000
    #./submit.sh gpu $task $common --gmm_prekappa0=20.0 --gmm_components=10000 --initial_weights=output/early_mix/--gmm_prekappa0=20.0/model.ckpt-370000
    #./submit.sh gpu $task $common --gmm_prekappa0=30.0 --gmm_components=10000 --initial_weights=output/early_mix/--gmm_prekappa0=30.0/model.ckpt-370000
#
    #./submit.sh gpu $task $common --gmm_prekappa0=20.0 --gmm_components=100000 --initial_weights=output/early_mix/--gmm_prekappa0=20.0/model.ckpt-370000
    #./submit.sh gpu $task $common --gmm_prekappa0=20.0 --gmm_components=20000 --initial_weights=output/early_mix/--gmm_prekappa0=20.0/model.ckpt-370000
    #./submit.sh gpu $task $common --gmm_prekappa0=20.0 --gmm_components=50000 --initial_weights=output/early_mix/--gmm_prekappa0=20.0/model.ckpt-370000
#
    ./submit.sh gpu $task $common --gmm_prekappa0=20.0 --gmm_components=100000 --initial_weights=output/early_mix/--gmm_prekappa0=20.0/model.ckpt-370000
    ./submit.sh gpu $task $common --gmm_prekappa0=20.0 --gmm_components=10000 --initial_weights=output/early_mix/--gmm_prekappa0=20.0/model.ckpt-370000

################################################################################
elif [[ $task = prekappa ]]; then

    common="--data=data/twitter-early --initial_weights=output/best/early-cnn/model.ckpt-520000 --pos_type=aglm_mix --gmm_type=verysimple --gmm_components=10000 --seed=1 --learningrate=1e-4"

    ./submit.sh gpu $task $common --gmm_prekappa0=0.1
    ./submit.sh gpu $task $common --gmm_prekappa0=1.0
    ./submit.sh gpu $task $common --gmm_prekappa0=5.0
    ./submit.sh gpu $task $common --gmm_prekappa0=10.0
    ./submit.sh gpu $task $common --gmm_prekappa0=20.0
    ./submit.sh gpu $task $common --gmm_prekappa0=30.0

################################################################################
elif [[ $task = dist ]]; then

    common="--data=data/twitter-early --initial_weights=output/best/early-cnn/model.ckpt-520000 --pos_type=aglm_mix --gmm_components=10000 --seed=1 --learningrate=1e-4"

    #./submit.sh gpu $task $common --gmm_type=verysimple --gmm_prekappa0=8.0 --gmm_distribution=fvm
    #./submit.sh gpu $task $common --gmm_type=verysimple --gmm_prekappa0=10.0 --gmm_distribution=fvm
    #./submit.sh gpu $task $common --gmm_type=verysimple --gmm_prekappa0=12.0 --gmm_distribution=fvm
    #./submit.sh gpu $task $common --gmm_type=verysimple --gmm_prekappa0=14.0 --gmm_distribution=fvm

    ./submit.sh gpu $task $common --gmm_type=verysimple --gmm_prekappa0=10.0 --gmm_distribution=efam
    ./submit.sh gpu $task $common --gmm_type=verysimple --gmm_prekappa0=0.1 --gmm_distribution=efam

    #./submit.sh gpu $task $common --gmm_type=verysimple --gmm_prekappa0=10.0 --gmm_distribution=gaussian
    #./submit.sh gpu $task $common --gmm_type=verysimple --gmm_prekappa0=12.0 --gmm_distribution=gaussian
    #./submit.sh gpu $task $common --gmm_type=verysimple --gmm_prekappa0=15.0 --gmm_distribution=gaussian
    #./submit.sh gpu $task $common --gmm_type=simple --gmm_prekappa0=10.0 --gmm_distribution=gaussian

################################################################################
elif [[ $task = mix ]]; then

    common="--data=data/twitter-early --initial_weights=output/best/early-cnn/model.ckpt-520000 --pos_type=aglm_mix --learningrate=1e-5"

    ./submit.sh gpu $task $common --gmm_type=verysimple --gmm_components=10000
    ./submit.sh gpu $task $common --gmm_type=verysimple --gmm_components=100000
    ./submit.sh gpu $task $common --gmm_type=simple --gmm_components=10000
    ./submit.sh gpu $task $common --gmm_type=simple --gmm_components=100000
    ./submit.sh gpu $task $common --gmm_type=simple --gmm_components=10000  --gmm_distloss
    ./submit.sh gpu $task $common --gmm_type=simple --gmm_components=100000 --gmm_distloss
    ./submit.sh gpu $task $common --gmm_type=complex --gmm_components=100
    ./submit.sh gpu $task $common --gmm_type=complex --gmm_components=100 --gmm_distloss

################################################################################
elif [[ $task = mix2 ]]; then
    task=mix

    common="--data=data/twitter-early --initial_weights=output/best/early-cnn-mix10000/model.ckpt-30000 --pos_type=aglm_mix --learningrate=1e-5"

    ./submit.sh gpu $task $common --gmm_type=simple --gmm_components=10000
    ./submit.sh gpu $task $common --gmm_type=simple --gmm_components=10000 --gmm_distloss

    common="--data=data/twitter-early --initial_weights=output/best/early-cnn-mix100000/model.ckpt-20000 --pos_type=aglm_mix --learningrate=1e-5"

    ./submit.sh gpu $task $common --gmm_type=simple --gmm_components=100000
    ./submit.sh gpu $task $common --gmm_type=simple --gmm_components=100000 --gmm_distloss

    common="--data=data/twitter-early --initial_weights=output/best/early-cnn-mix100000/model.ckpt-20000 --pos_type=aglm_mix --learningrate=5e-6"

    ./submit.sh gpu $task $common --gmm_type=simple --gmm_components=100000
    ./submit.sh gpu $task $common --gmm_type=simple --gmm_components=100000 --gmm_distloss

################################################################################
elif [[ $task = lrfactor ]]; then
    task=mix

    common="--data=data/twitter-early --initial_weights=output/best/early-cnn-mix10000/model.ckpt-30000 --pos_type=aglm_mix --learningrate=1e-5"

    ./submit.sh gpu $task $common --gmm_type=simple --gmm_components=10000 --gmm_lrfactor=1e-7
    ./submit.sh gpu $task $common --gmm_type=simple --gmm_components=10000 --gmm_lrfactor=1e-8
    ./submit.sh gpu $task $common --gmm_type=simple --gmm_components=10000 --gmm_lrfactor=1e-9

    common="--data=data/twitter-early --initial_weights=output/best/early-cnn-mix100000/model.ckpt-20000 --pos_type=aglm_mix --learningrate=1e-5"

    ./submit.sh gpu $task $common --gmm_type=simple --gmm_components=100000 --gmm_lrfactor=1e-7
    ./submit.sh gpu $task $common --gmm_type=simple --gmm_components=100000 --gmm_lrfactor=1e-8
    ./submit.sh gpu $task $common --gmm_type=simple --gmm_components=100000 --gmm_lrfactor=1e-9

################################################################################
fi
