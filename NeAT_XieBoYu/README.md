## NeAT模型复现

工程结构如下：
```aiignore
|-dataset          - 存放数据集
    |-iawe_1
        |-test.txt
        |-train.txt
        |-valid.txt
|-model
    |-metrics.py   - 评价指标
    |-NeAT.py      - NeAT模型
|-result           - 存放运行结果
|-utils
    |-load_data.py - 读取数据集
|-main.py          - 运行模型
```

Paper: Ahn D, Saini U S, Papalexakis E E, et al. Neural additive tensor decomposition for sparse tensors[C]//Proceedings of the 33rd ACM International Conference on Information and Knowledge Management. 2024: 14-23.
[paper link](https://doi.org/10.1145/3627673.3679833)