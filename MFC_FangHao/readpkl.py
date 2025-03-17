import networkx as nx
import pickle
#
# 用二进制读取模式打开.pkl文件
with open('Data/high2011.pkl', 'rb') as file:
    data = pickle.load(file)

# 打印加载的数据
print(data)