# Trained-Rank-Pruning
Paper has been accepted by IJCAI2020.
PyTorch code demo for ["Trained Rank Pruning for Efficient Deep Neural Networks"](https://arxiv.org/abs/2004.14566v1)<br>
Our code is built based on  [bearpaw](https://github.com/bearpaw/pytorch-classification)<br>
<img src=framework.png width=80% align="middle"><br>
What's in this repo so far:
 * TRP code for CIFAR-10 experiments
 * Nuclear regularization code for CIFAR-10 experiments

# prerequisite
* torch
* numpy
* matplotlib
* progress
 
#### Simple Examples
```Shell
optional arguments:
  -a                    model_name
  --depth               number layers
  --epoths              training epochs
  -c                    path to save checkpoints
  --gpu-id              specifiy using GPU or not
  --nuclear-weight      nuclear regularization weight (if not set, nuclear  reglularization is not used)
  --trp                 boolean value, set to enable TRP training
  --type                the decompsition type 'NC','VH','ND'
```
Training ResNet-20 baseline:

```
python cifar-TRP.py -a resnet --depth 20 --epochs 164 --schedule 81 122 --gamma 0.1 --wd 1e-4 --checkpoint checkpoints/cifar10/resnet-20 

```
Training ResNet-20 with nuclear norm:

```
python cifar-TRP.py -a resnet --depth 20 --epochs 164 --schedule 81 122 --gamma 0.1 --wd 1e-4 --checkpoint checkpoints/cifar10/resnet-20 --nuclear-weight 0.0003

```
Training ResNet-20 with TRP and nuclear norm:
```
python cifar-TRP.py -a resnet --depth 20 --epochs 164 --schedule 81 122 --gamma 0.1 --wd 1e-4 --checkpoint checkpoints/cifar10/resnet-20 --nuclear-weight 0.0003 --trp --type NC

```
Decompose the trained model without retraining:
```
python cifar-nuclear-regularization.py.py -a resnet --depth 20 --resume checkpoints/cifar10/resnet-20/model_best.pth.tar --evaluate --type NC

```
Decompose the trained model with retraining:
```
python cifar-nuclear-regularization.py.py -a resnet --depth 20 --resume checkpoints/cifar10/resnet-20/model_best.pth.tar --evaluate --type NC --retrain

```

#### Notes
During decomposition, TRP using value threshold(very small value to truncate singular values) based strategy because the trained model is in low-rank format. Other methods including Channel or spatial-wise decomposition baseline methods use energy threshold.
## Results

- Results on CIFAR-10(without decomposing the final FC):

|Network| Method |  Scheme  | # Params | FLOPs |Acc|
|:-----|:-------:|:-----:|:--------:|:-----:|:-----:|
|Resnet20| Origin |   None   | 0.27M | 1x    |91.74|
|Resnet20| TRP+Nu |Channel| 0.1M | 2.17x|90.50|
|Resnet20| TRP+Nu |Spatial| 0.08M | 2.84x |90.62|
|Resnet20| TRP+Nu | ND | 0.14M | 2.04x | 90.88 |
|Resnet32| Origin |   None   | 0.47M | 1x    |92.26|
|Resnet32| TRP+Nu |Channel| 0.16M | 2.2x|91.40|
|Resnet32| TRP+Nu |Spatial| 0.11M | 3.4x |91.39|

- Results on ImageNet(without decomposing the final FC):

|Network| Method |  Scheme  | FLOPs |Top1|Top5|
|:-----|:-------:|:-----:|:-----:|:-----:|:-----:|
|Resnet50| Origin |  None | 1x    |75.90|92.80|
|Resnet50| TRP+Nu |Channel| 2.23x|72.69|91.41|
|Resnet50| TRP+Nu |Channel| 1.80x|74.06|92.07|
|Resnet50| Channel Pruning(ICCV) |None|2.00x|-|90.91|
|Resnet50| Filter Pruning(ICCV) |None|1.58x|72.04|90.67|
|Resnet50| Filter Pruning(TPAMI) |None|2.26x|72.03|90.99|
### Citation
If you think this work is helpful for your own research, please consider add following bibtex config in your latex file

```Latex
@article{xu2018trained,
  title={Trained Rank Pruning for Efficient Deep Neural Networks},
  author={Xu, Yuhui and Li, Yuxi and Zhang, Shuai and Wen, Wei and Wang, Botao and Qi, Yingyong and Chen, Yiran and Lin, Weiyao and Xiong, Hongkai},
  journal={arXiv preprint arXiv:1812.02402},
  year={2018}
}
