#!/bin/bash
python3 cifar-TRP.py -a resnet --depth 20 --epochs 128 --schedule 81 112 --gamma 0.1 --wd 1e-4 --checkpoint checkpoint/cifar10/resnet-20 --type ND -dp 5 -d cifar10 --gpu-id 1 --trp
