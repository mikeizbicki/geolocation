set -e
task='superbig'

################################################################################
if [[ $task = superbig ]]; then

    common='--data=data/BillionTwitter/ --initial_weights=output/best/cnn-med/ --train_list output/pos --learningrate=1e-5 --decay=1e5 --gmm_decomposed 1000 --gmm_components=500000 --gmm_prekappa0=12.0'
    ./submit.sh gpu $task $common --gmm_sparsity=10000
    ./submit.sh gpu $task $common

################################################################################
elif [[ $task = fullrun_basic ]]; then

    common='--data=data/BillionTwitter --output pos country --summary_newusers'
    ./submit.sh cpu $task $common --input lang --full --learningrate=9.10e-03
    ./submit.sh cpu $task $common --input time --full --learningrate=2.32e-03
    ./submit.sh cpu $task $common --input lang time --full 256 256 --learningrate=9.82e-03 --l2=1.17e-04

################################################################################
elif [[ $task = fullrun2 ]]; then

    ./submit.sh gpu6 $task --data=data/BillionTwitter --initial_weights=output/best/early-cnn-large --summary_size=med --summary_newusers
    ./submit.sh gpu $task --data=data/BillionTwitter --initial_weights=output/best/early-cnn-small --summary_size=med --summary_newusers

    #./submit.sh gpu2 $task --data=data/twitter-full --initial_weights=output/early.old/--data=data,twitter-early,--predict_lang=True,--batchsize=100,--learningrate=5e-4,--decay=1e5,--text_naive=False,--pos_type=aglm_mix --l1=2
    #./submit.sh gpu $task --data=data/twitter-full --initial_weights=output/early.old/--data=data,twitter-early,--predict_lang=True,--batchsize=100,--learningrate=5e-4,--decay=1e5,--text_naive=False,--pos_type=aglm_mix --l1=0

    #./submit.sh gpu2 $task --data=data/twitter-full --initial_weights=output/early.old/--data=data,twitter-early,--predict_lang=True,--batchsize=100,--learningrate=5e-4,--decay=1e5,--text_naive=False,--pos_type=aglm_mix --l1=2
    #./submit.sh gpu2 $task --data=data/twitter-full --initial_weights=output/early.old/--data=data,twitter-early,--predict_lang=True,--batchsize=100,--learningrate=5e-4,--decay=1e5,--text_naive=False,--pos_type=aglm_mix --l1=2 --summary_newusers
    #./submit.sh gpu2 $task --data=data/twitter-full --initial_weights=output/early.old/--data=data,twitter-early,--predict_lang=True,--batchsize=100,--learningrate=5e-4,--decay=1e5,--text_naive=False,--pos_type=aglm_mix --l1=2 --summary_newusers --summary_size=all

    #./submit.sh gpu $task --data=data/twitter-full --initial_weights=output/early.old/--data=data,twitter-early,--predict_lang=True,--batchsize=100,--learningrate=5e-4,--decay=1e5,--text_naive=False,--pos_type=aglm_mix --l1=0 --summary_newusers
    #./submit.sh gpu $task --data=data/twitter-full --initial_weights=output/early.old/--data=data,twitter-early,--predict_lang=True,--batchsize=100,--learningrate=5e-4,--decay=1e5,--text_naive=False,--pos_type=aglm_mix --l1=0 --summary_size=all
    #./submit.sh gpu $task --data=data/twitter-full --initial_weights=output/early.old/--data=data,twitter-early,--predict_lang=True,--batchsize=100,--learningrate=5e-4,--decay=1e5,--text_naive=False,--pos_type=aglm_mix --l1=0 --summary_size=all --summary_newusers

################################################################################
elif [[ $task = early ]]; then

    #common="--data=data/twitter-early --predict_lang=True --batchsize=100 --learningrate=5e-4 --decay=1e5"
#
    #./submit.sh gpu $task $common --text_naive=False --pos_type=naive
    #./submit.sh gpu $task $common --text_naive=False --pos_type=aglm
    #./submit.sh gpu $task $common --text_naive=False --pos_type=aglm_mix
    #./submit.sh gpu $task $common --text_naive=True --pos_type=aglm_mix --input cnn lang
    #./submit.sh gpu $task $common --text_naive=True --pos_type=aglm_mix --input cnn lang --predict_lang_use=True
    #./submit.sh gpu $task $common --text_naive=False --pos_type=aglm_mix --full 4096 4096 --cltcc_numfilters=2048 --cnn_vocabsize=256
    #./submit.sh gpu $task $common --text_naive=False --pos_type=aglm_mix --full 1024 1024 --cltcc_numfilters=512

    common="--data=data/twitter-early --predict_lang=True --batchsize=100 --learningrate=5e-4 --decay=1e5"
    ./submit.sh gpu $task $common --text_naive=False --pos_type=aglm_mix --full 1024 1024 --cltcc_numfilters=512 --full_per_lang

    #./submit.sh gpu $task $common --text_naive=False --pos_type=aglm_mix --full 1024 1024 --cltcc_numfilters=512
    #./submit.sh gpu $task $common --text_naive=False --learningrate=2e-3
    #./submit.sh gpu $task $common --text_naive=False --learningrate=1e-3
    #./submit.sh gpu $task $common --text_naive=False --learningrate=5e-4

    #common="--data=data/twitter-early --decay=1e5 --learningrate=5e-4 --predict_lang=True --pos_type=aglm"
    #./submit.sh gpu $task $common --text_naive=False --text_transliterate=True --text_latin_bit=True --text_multichar_init_bit=True

elif [[ $task = early2 ]]; then

    ./submit.sh gpu $task --decay=1e5 --initial_weights=output/early.old/--data=data,twitter-early,--predict_lang=True,--batchsize=100,--learningrate=5e-4,--text_naive=False,--pos_type=aglm
    ./submit.sh gpu $task --decay=1e5 --initial_weights=output/early.old/--data=data,twitter-early,--predict_lang=True,--batchsize=100,--learningrate=5e-4,--text_naive=False,--pos_type=aglm_mix
    ./submit.sh gpu $task --decay=1e5 --initial_weights=output/early.old/--data=data,twitter-early,--predict_lang=True,--batchsize=100,--learningrate=5e-4,--text_naive=False,--pos_type=aglm_mix,--full,1024,1024,--cltcc_numfilters=512
    ./submit.sh gpu $task --decay=1e5 --initial_weights=output/early.old/--data=data,twitter-early,--predict_lang=True,--batchsize=100,--learningrate=5e-4,--text_naive=False,--pos_type=aglm_mix,--full,4096,4096,--cltcc_numfilters=2048,--cnn_vocabsize=256
    ./submit.sh gpu $task --decay=1e5 --initial_weights=output/early.old/--data=data,twitter-early,--predict_lang=True,--batchsize=100,--learningrate=5e-4,--text_naive=False,--pos_type=naive
    ./submit.sh gpu $task --decay=1e5 --initial_weights=output/early.old/--data=data,twitter-early,--predict_lang=True,--batchsize=100,--learningrate=5e-4,--text_naive=True,--pos_type=aglm_mix,--input,cnn,lang
    ./submit.sh gpu $task --decay=1e5 --initial_weights=output/early.old/--data=data,twitter-early,--predict_lang=True,--batchsize=100,--learningrate=5e-4,--text_naive=True,--pos_type=aglm_mix,--input,cnn,lang,--predict_lang_use=True
    ./submit.sh gpu $task --decay=1e5 --initial_weights=output/early.old/--data=data,twitter-early.es,--predict_lang=True,--batchsize=100,--learningrate=5e-4,--text_naive=False,--pos_type=aglm_mix,--full,1024,1024,--cltcc_numfilters=512

################################################################################
elif [[ $task = decompose ]]; then
    common="--data=data/twitter-early --input cnn lang time --full 2048 2048 --output country pos --pos_type=aglm_mix --gmm_type=verysimple --learningrate=5e-4 --decay=1e5 --gmm_prekappa0=10.0 --gmm_components=10000"

    ./submit.sh gpu $task $common --gmm_decompose 16
    ./submit.sh gpu $task $common --gmm_decompose 16 256
    ./submit.sh gpu $task $common --gmm_decompose 16 256 4096
    ./submit.sh gpu $task $common --gmm_decompose 256
    ./submit.sh gpu $task $common --gmm_decompose 32 1024

################################################################################
elif [[ $task = lr2 ]]; then

    common="--data=data/twitter-early --pos_type=aglm_mix --gmm_type=verysimple --gmm_components=10000 --decay=1e5"

    #./submit.sh gpu $task $common --cnn_type=vdcnn --learningrate=1e-3
    #./submit.sh gpu $task $common --cnn_type=vdcnn --learningrate=5e-3
    #./submit.sh gpu $task $common --cnn_type=vdcnn --learningrate=1e-2

    #./submit.sh gpu $task $common --cnn_type=cltcc --learningrate=1e-3
    #./submit.sh gpu $task $common --cnn_type=cltcc --learningrate=5e-3
    #./submit.sh gpu $task $common --cnn_type=cltcc --learningrate=1e-2

    common="--data=data/twitter-early --pos_type=aglm --decay=1e5"

    ./submit.sh gpu $task $common --cnn_type=vdcnn --vdcnn_size=0 --learningrate=5e-4
    ./submit.sh gpu $task $common --cnn_type=vdcnn --vdcnn_size=1 --learningrate=5e-4
    ./submit.sh gpu $task $common --cnn_type=vdcnn --vdcnn_size=2 --learningrate=5e-4
    ./submit.sh gpu $task $common --cnn_type=vdcnn --vdcnn_size=3 --learningrate=5e-4
    #./submit.sh gpu $task $common --cnn_type=cltcc --learningrate=5e-4

################################################################################
elif [[ $task = newhash2 ]]; then

    common="--data=data/twitter-early --pos_type=aglm --decay=1e5 --learningrate=5e-4"

    #./submit.sh gpu $task $common --text_naive=True --cnn_type=vdcnn --vdcnn_size=0 --learningrate=5e-4
    #./submit.sh gpu $task $common --text_naive=True --cnn_type=vdcnn --vdcnn_size=1 --learningrate=5e-4
    #./submit.sh gpu $task $common --text_naive=True --cnn_type=vdcnn --vdcnn_size=2 --learningrate=5e-4
    #./submit.sh gpu $task $common --text_naive=True --cnn_type=vdcnn --vdcnn_size=3 --learningrate=5e-4
    #./submit.sh gpu $task $common --text_naive=True --cnn_type=cltcc --learningrate=5e-4

    #./submit.sh gpu $task $common --text_naive=False --text_transliterate=True --cnn_vocabsize=192 --cnn_type=cltcc --learningrate=5e-4
    #./submit.sh gpu $task $common --text_naive=False --text_transliterate=True --cnn_vocabsize=192 --cnn_type=vdcnn --vdcnn_size=1 --learningrate=5e-4
    #./submit.sh gpu $task $common --text_naive=False --text_transliterate=True --cnn_vocabsize=192 --cnn_type=vdcnn --vdcnn_size=2 --learningrate=5e-4

    #./submit.sh gpu $task $common --text_naive=False --text_transliterate=False --cnn_vocabsize=192 --cnn_type=cltcc --learningrate=5e-4
    #./submit.sh gpu $task $common --text_naive=False --text_transliterate=False --cnn_vocabsize=192 --cnn_type=vdcnn --vdcnn_size=1 --learningrate=5e-4
    #./submit.sh gpu $task $common --text_naive=False --text_transliterate=False --cnn_vocabsize=192 --cnn_type=vdcnn --vdcnn_size=2 --learningrate=5e-4

    #common="--data=data/twitter-early --pos_type=aglm_mix --gmm_type=verysimple --gmm_components=10000 --decay=1e5 --learningrate=5e-4"

    common="--data=data/twitter-early --decay=1e5 --learningrate=5e-4 --predict_lang=True --pos_type=aglm_mix"

    ./submit.sh gpu $task $common --text_naive=True
    ./submit.sh gpu $task $common --text_naive=True --predict_lang_use=True
    ./submit.sh gpu $task $common --text_naive=False --text_transliterate=False
    ./submit.sh gpu $task $common --text_naive=False --text_transliterate=True --text_latin_bit=False --text_multichar_init_bit=False
    ./submit.sh gpu $task $common --text_naive=False --text_transliterate=True --text_latin_bit=True --text_multichar_init_bit=True
    ./submit.sh gpu $task $common --text_naive=False --text_transliterate=True --text_latin_bit=True --text_multichar_init_bit=True --predict_lang_use=True
    ./submit.sh gpu $task $common --text_naive=False --text_transliterate=True --text_latin_bit=True --text_multichar_init_bit=True --predict_lang_use=True --gmm_distloss

    common="--data=data/twitter-early --decay=1e5 --learningrate=5e-4 --predict_lang=True --pos_type=aglm"

    ./submit.sh gpu $task $common --text_naive=False --text_transliterate=True --text_latin_bit=True --text_multichar_init_bit=True

################################################################################
elif [[ $task = newhash ]]; then

    common="--data=data/twitter-early --pos_type=aglm_mix --gmm_type=verysimple --gmm_components=10000 --decay=1e5 --learningrate=5e-4"

    ./submit.sh gpu $task $common --cnn_type=vdcnn --text_naive=True
    ./submit.sh gpu $task $common --cnn_type=vdcnn --text_naive=False --text_transliterate=False
    ./submit.sh gpu $task $common --cnn_type=vdcnn --text_naive=False --text_transliterate=True

    ./submit.sh gpu $task $common --text_naive=True
    ./submit.sh gpu $task $common --text_naive=False --text_transliterate=False
    ./submit.sh gpu $task $common --text_naive=False --text_transliterate=True --text_latin_bit=False --text_multichar_init_bit=False
    ./submit.sh gpu $task $common --text_naive=False --text_transliterate=True --text_latin_bit=True --text_multichar_init_bit=True
    ./submit.sh gpu $task $common --text_naive=False --text_transliterate=True --text_latin_bit=True --text_multichar_init_bit=True --cnn_vocabsize=130

################################################################################
elif [[ $task = bigrun ]]; then

    common="--data=data/twitter --initial_weights=output/best/early-cnn-dist8/model.ckpt-140000 --pos_type=aglm_mix --gmm_components=10000 --summary_size=med --decay=5e5"

    #./submit.sh gpu $task $common --gmm_type=verysimple --learningrate=5e-5 --data_style=online
    #./submit.sh gpu $task $common --gmm_type=verysimple --learningrate=5e-5 --data_style=batch

    #./submit.sh gpu $task $common --gmm_type=verysimple --learningrate=1e-5 --data_style=online
    #./submit.sh gpu $task $common --gmm_type=verysimple --learningrate=1e-5 --data_style=batch

    #./submit.sh gpu $task $common --gmm_type=verysimple --learningrate=1e-5 --data_style=online --predict_lang_use
    #./submit.sh gpu $task $common --gmm_type=verysimple --learningrate=1e-5 --data_style=batch --predict_lang_use

    common="--data=data/twitter --initial_weights=output/best/early-cnn-dist10/model.ckpt-140000 --pos_type=aglm_mix --gmm_components=10000 --summary_size=med --decay=5e5"

    #./submit.sh gpu $task $common --gmm_type=verysimple --learningrate=5e-5 --data_style=online
    #./submit.sh gpu $task $common --gmm_type=verysimple --learningrate=5e-5 --data_style=batch

    common="--data=data/twitter --pos_type=aglm_mix --gmm_components=10000 --summary_size=med --decay=1e5 --dropout=0.5"
    ./submit.sh gpu $task $common --gmm_type=verysimple --learningrate=5e-4 --initial_weights=output/bigrun.warm/8.5e-5.batch2/model.ckpt-30000
    ./submit.sh gpu $task $common --gmm_type=verysimple --learningrate=5e-4 --initial_weights=output/bigrun.warm/10.5e-5.batch2/model.ckpt-30000
    ./submit.sh gpu $task $common --gmm_type=verysimple --learningrate=1e-4 --initial_weights=output/bigrun.warm/8.1e-5.batch2/model.ckpt-30000
    ./submit.sh gpu $task $common --gmm_type=verysimple --learningrate=1e-4 --initial_weights=output/bigrun.warm/8.1e-5.batch.lang/model.ckpt-30000 --predict_lang_use


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

    ./submit.sh gpu $task $common --gmm_type=verysimple --gmm_prekappa0=6.0 --gmm_distribution=fvm
    #./submit.sh gpu $task $common --gmm_type=verysimple --gmm_prekappa0=8.0 --gmm_distribution=fvm
    ./submit.sh gpu $task $common --gmm_type=verysimple --gmm_prekappa0=9.0 --gmm_distribution=fvm
    #./submit.sh gpu $task $common --gmm_type=verysimple --gmm_prekappa0=10.0 --gmm_distribution=fvm
    #./submit.sh gpu $task $common --gmm_type=verysimple --gmm_prekappa0=12.0 --gmm_distribution=fvm
    #./submit.sh gpu $task $common --gmm_type=verysimple --gmm_prekappa0=14.0 --gmm_distribution=fvm

    #./submit.sh gpu $task $common --gmm_type=verysimple --gmm_prekappa0=10.0 --gmm_distribution=efam
    #./submit.sh gpu $task $common --gmm_type=verysimple --gmm_prekappa0=0.1 --gmm_distribution=efam

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
