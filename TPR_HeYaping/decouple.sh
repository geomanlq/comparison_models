 #!/bin/bash
 python3 cifar-TRP.py --gpu-id 1 -d cifar10 --evaluate --resume checkpoint/cifar10/resnet-20/model_best.pth.tar --type ND --arch resnet --depth 20 --test-batch 25
