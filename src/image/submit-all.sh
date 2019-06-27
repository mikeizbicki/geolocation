sbatch src/image2/submit.sh --model=ResNet50 --batchsize=128 --learningrate=5e-4 --pretrain --train_only_last
sbatch src/image2/submit.sh --model=ResNet50 --batchsize=128 --learningrate=5e-4 --pretrain
sbatch src/image2/submit.sh --model=ResNet50 --batchsize=128 --learningrate=5e-4

sbatch src/image2/submit.sh --model=ResNet50 --batchsize=128 --learningrate=1e-3 --pretrain
sbatch src/image2/submit.sh --model=ResNet50 --batchsize=128 --learningrate=1e-4 --pretrain
sbatch src/image2/submit.sh --model=ResNet50 --batchsize=128 --learningrate=5e-5 --pretrain

#sbatch src/image2/train.py --inputs --model=ResNet50 --batchsize=128 --learningrate=5e-4
