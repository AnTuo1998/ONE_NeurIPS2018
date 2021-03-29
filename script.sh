CUDA_VISIBLE_DEVICES=$1 python cifar_one.py \
    --gpu-id $1 \
    --dataset cifar10 \
    --arch resnet18 \
    --manualSeed 0
