# import pickle
#
#
# # 定义一个函数来加载并检查 pkl 文件的内容
# def inspect_pkl(file_path):
#     # 打开 pkl 文件
#     with open(file_path, 'rb') as f:
#         # 加载文件内容
#         data = pickle.load(f)
#
#     # 打印对象类型
#     print(f"Type of data: {type(data)}")
#
#     # 检查不同类型对象的内容
#     if isinstance(data, dict):
#         print("Keys:", data.keys())
#         print("Sample value:", next(iter(data.values())))
#     elif isinstance(data, list):
#         print("Length of list:", len(data))
#         print("Sample item:", data[0])
#     else:
#         print("Data:", data)
#
#
# # 替换为你的 pkl 文件路径
# file_path = '../Data/dblp/trnMat.pkl'
# inspect_pkl(file_path)
######
import numpy as np
from scipy.sparse import coo_matrix
import pickle

# 读取txt文件，并解析数据
def read_txt_file(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split('\t')  # 假设是以tab分隔的txt文件
            if len(parts) >= 4:
                row = int(parts[0])  # 第一列作为稀疏矩阵的行索引
                col = int(parts[1])  # 第二列作为稀疏矩阵的列索引
                value = float(parts[3])  # 第四列作为值
                data.append((row, col, value))
    return data

# 示例：假设有一个名为data.txt的文件，格式为 row \t col \t other \t value
txt_file_path = '/cora/cdf-test90%.txt'
data = read_txt_file(txt_file_path)

# 解析数据并创建稀疏矩阵
rows = [item[0] for item in data]
cols = [item[1] for item in data]
values = [item[2] for item in data]

# 获取稀疏矩阵的维度
n_rows = max(rows) + 1
n_cols = max(cols) + 1
# 创建稀疏矩阵
sparse_matrix = coo_matrix((values, (rows, cols)), shape=(n_rows, n_cols))

# 保存为pkl文件
pkl_file_path = '/cora/tstMat.pkl'
with open(pkl_file_path, 'wb') as pkl_file:
    pickle.dump(sparse_matrix, pkl_file)

