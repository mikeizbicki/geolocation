#--log_dir=log --log_name=wnut_large

proj=wnut2016_2

#./scripts/submit.sh gpu $proj --data=data/wnut2016/training --output wnut2016 --train_list=wnut2016 --summary_size=small --initial_weights=models/unicodecnn-large/ --learningrate=5e-4 --wnut2016_biasonly

#./scripts/submit.sh gpu $proj --data=data/wnut2016/training --output wnut2016 --train_list=wnut2016 --summary_size=small --initial_weights=models/unicodecnn-large/ --learningrate=5e-4
#./scripts/submit.sh gpu $proj --data=data/wnut2016/training --output wnut2016 --train_list=wnut2016 --summary_size=small --initial_weights=models/unicodecnn-small/ --learningrate=5e-4
./scripts/submit.sh gpu $proj --data=data/wnut2016/training --output wnut2016 --summary_size=small --initial_weights=models/unicodecnn-large/ --learningrate=5e-4
./scripts/submit.sh gpu $proj --data=data/wnut2016/training --output wnut2016 --summary_size=small --initial_weights=models/unicodecnn-small/ --learningrate=5e-4

#./scripts/submit.sh gpu $proj --initial_weights=output/wnut2016_1/lastlayer_large --train_list=wnut2016
#./scripts/submit.sh gpu $proj --initial_weights=output/wnut2016_1/lastlayer_small --train_list=wnut2016
#./scripts/submit.sh gpu $proj --initial_weights=output/wnut2016_1/lastlayer_large --train_list
#./scripts/submit.sh gpu $proj --initial_weights=output/wnut2016_1/lastlayer_small --train_list
