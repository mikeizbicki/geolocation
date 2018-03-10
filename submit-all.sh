#!/bin/bash

echo "lang time"
for i in $(seq 1 20); do
    learningrate=$(python -c "import random;print(10**random.uniform(-5.0, .0))")
    l2=$(python -c "import random;print(10**random.uniform(-6.0, 0.0))")
    cmd="--data=data/twitter-small/ --input lang time --full 256 256 --output=aglm --loss=dist --learningrate=$learningrate --l2=$l2"
    ./submit.sh cpu $cmd
done

echo "lang time bow"
for i in $(seq 1 20); do
    learningrate=$(python -c "import random;print(10**random.uniform(-5.0, 0.0))")
    l2=$(python -c "import random;print(10**random.uniform(-6.0, 0.0))")
    l1=$(python -c "import random;print(10**random.uniform(-6.0, 0.0))")
    cmd="--data=data/twitter-small/ --input lang time bow --full 256 256 --output=aglm --loss=dist --learningrate=$learningrate --l2=$l2 --l1=$l1"
    ./submit.sh cpu $cmd
done

echo "lang time bow cltcc"
for i in $(seq 1 6); do
    learningrate=$(python -c "import random;print(10**random.uniform(-5.0, -3.0))")
    l2=$(python -c "import random;print(10**random.uniform(-6.0, -3.0))")
    l1=$(python -c "import random;print(10**random.uniform(-6.0, -3.0))")
    cmd="--data=data/twitter-small/ --input lang time bow cltcc --full 256 256 --output=aglm --loss=dist --learningrate=$learningrate --l2=$l2 --l1=$l1"
    ./submit.sh gpu $cmd
done
