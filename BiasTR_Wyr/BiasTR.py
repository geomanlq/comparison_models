import time
from numba.experimental import jitclass
import numpy as np
from numba.typed import List
import pandas as pd
from numba import jit, int32, float64, njit
from itertools import product
rank=20
minMAERound=0
minRMSERound=0
minRound=0
maxValue = 0
minValue = np.inf
trainCount = 0
testCount = 0
validCount = 0
trainRound = 200
spec = [('aID', int32),
        ('bID', int32),
        ('cID', int32),
        ('value', float64),
        ('valueHat', float64)]
@jitclass(spec)
class TensorTuple(object):
    def __init__(self):
        self.aID = 0
        self.bID = 0
        self.cID = 0
        self.value = 0.0
        self.valueHat = 0.0

def initData(inputFile1, inputFile2,inputFile3,separator):
    global maxAID, maxBID, maxCID, minAID, minBID, minCID, minValue, maxValue, trainCount, testCount, validCount
    data = List()
    start_time = time.time()
    df1 = pd.read_csv(inputFile1, sep=separator, names=['aID', 'bID', 'cID', 'value'],
                     dtype={'aID': int, 'bID': int, 'cID': int, 'value': float}, engine='python')
    df2 = pd.read_csv(inputFile2, sep=separator, names=['aID', 'bID', 'cID', 'value'],
                      dtype={'aID': int, 'bID': int, 'cID': int, 'value': float}, engine='python')
    df3 = pd.read_csv(inputFile3, sep=separator, names=['aID', 'bID', 'cID', 'value'],
                      dtype={'aID': int, 'bID': int, 'cID': int, 'value': float}, engine='python')
    df1.iloc[:, :3] = df1.iloc[:, :3].astype(int)
    df2.iloc[:, :3] = df2.iloc[:, :3].astype(int)
    df3.iloc[:, :3] = df3.iloc[:, :3].astype(int)
    maxAID = df1['aID'].max()
    maxBID = df1['bID'].max()
    maxCID = df1['cID'].max()
    minAID = df1['aID'].min()
    minBID = df1['bID'].min()
    minCID = df1['cID'].min()
    # 划分数据集的比列
    def dataframe_to_list(dataframe):
        data = List()
        for row in dataframe.itertuples(index=False):
            qtemp = TensorTuple()
            qtemp.aID = row.aID
            qtemp.bID = row.bID
            qtemp.cID = row.cID
            qtemp.value = row.value
            data.append(qtemp)
        return data
    trainData = dataframe_to_list(df1)
    testData = dataframe_to_list(df2)
    validData= dataframe_to_list(df3)
    end_time = time.time()
    iteration_time = end_time - start_time
    print("读取数据的时间：", iteration_time)
    return trainData,validData,testData, maxAID, maxBID, maxCID, minAID, minBID, minCID, df1.shape[0], df2.shape[0], df3.shape[0]

@jit(nopython=True)
def get_prediction(S,D,T,a,b,c,i,j,k):
    y_hat = [[0 for _ in range(rank)] for _ in range(rank)]
    result = [[0 for _ in range(rank)] for _ in range(rank)]
    tmnk = 0
    for r1 in range(rank):
        for r3 in range(rank):
            for r2 in range(rank):
                y_hat[r1][r3] += S[r1][i][r2] * D[r2][j][r3]
    for r1 in range(rank):
        for r1_prime in range(rank):
            for r3 in range(rank):
                result[r1][r1_prime] += y_hat[r1][r3] * T[r3][k][r1_prime]

    for r1 in range(rank):
        tmnk += result[r1][r1]
    tmnk+=a[i]+b[j]+c[k]
    return tmnk

@jit(nopython=True)
def LF_matric():
    initscale = 0.05
    S = np.random.rand(rank,maxAID+1,rank) * initscale
    D = np.random.rand(rank,maxBID+1,rank) * initscale
    T = np.random.rand(rank,maxCID+1,rank) * initscale
    a = np.random.rand(maxAID + 1) * initscale
    b = np.random.rand(maxBID + 1) * initscale
    c = np.random.rand(maxCID + 1) * initscale
    return S,D,T,a,b,c

@jit(nopython=True)
def Help_matric():
    Sup =np.zeros((rank,maxAID+1,rank))
    Sdown =np.zeros((rank,maxAID+1,rank))
    Dup =np.zeros((rank,maxBID+1,rank))
    Ddown =np.zeros((rank,maxBID+1,rank))
    Tup =np.zeros((rank,maxCID+1,rank))
    Tdown =np.zeros((rank,maxCID+1,rank))

    aup =np.zeros(maxAID+1)
    adown =np.zeros(maxAID+1)
    bup =np.zeros(maxBID+1)
    bdown =np.zeros(maxBID+1)
    cup =np.zeros(maxCID+1)
    cdown =np.zeros(maxCID+1)
    return Sup,Sdown,Dup,Ddown,Tup,Tdown,aup,adown,bup,bdown,cup,cdown


def grid_search(parameters):
    parameter_combinations = product(*parameters.values())
    return parameter_combinations

@jit(nopython=True)
def train(trainData, testData, validData,lambda_,yita):
    # 设置的误差阈值
    errorgap = 1E-4
    flag_rmse = True
    flag_mae = True
    S,D,T,a,b,c=LF_matric()
    # 连续下降轮数小于误差范围切达到阈值终止训练
    threshold=2
    minRMSE = 200.0
    minMAE = 200.0
    everyRoundRMSE = [0.0] * (trainRound + 1)
    everyRoundMAE = [0.0] * (trainRound + 1)
    everyRoundRMSE[0] = minRMSE
    everyRoundMAE[0] = minMAE
    everyRoundRMSE2 = [0.0] * (trainRound + 1)
    everyRoundMAE2 = [0.0] * (trainRound + 1)
    everyRoundRMSE2[0] = minRMSE
    everyRoundMAE2[0] = minMAE
    # 使用训练集进行训练
    for t in range(1, trainRound + 1):
        for train_tuple in trainData:
            i = train_tuple.aID
            j = train_tuple.bID
            k = train_tuple.cID
            train_tuple.valueHat = get_prediction(S, D, T, a, b, c, i, j, k)
            error = train_tuple.value - train_tuple.valueHat
            for r1 in range(rank):
                for r2 in range(rank):
                    grad_s = 0.0
                    for r3 in range(rank):
                        grad_s += error * D[r2][j][r3] * T[r3][k][r1]
                    S[r1][i][r2] += yita * (grad_s - lambda_ * S[r1][i][r2])
            for r2 in range(rank):
                for r3 in range(rank):
                    grad_d = 0.0
                    for r1 in range(rank):
                        grad_d += error * T[r3][k][r1] * S[r1][i][r2]
            for r3 in range(rank):
                for r1 in range(rank):
                    grad_t = 0.0
                    for r2 in range(rank):
                        grad_t += error * S[r1][i][r2] * D[r2][j][r3]
                    T[r3][k][r1] += yita * (grad_t - lambda_ * T[r3][k][r1])
            a[i] = a[i] + yita * (error - lambda_ * a[i])
            b[j] = b[j] + yita * (error - lambda_ * b[j])
            c[k] = c[k] + yita * (error - lambda_ * c[k])
        square = 0.0
        abs_count = 0.0
        for test_tuple in testData:
            i = test_tuple.aID
            j = test_tuple.bID
            k = test_tuple.cID
            test_tuple.valueHat = get_prediction(S,D,T, a, b, c, i, j, k)
            square += (test_tuple.value - test_tuple.valueHat) ** 2
            abs_count += abs(test_tuple.value - test_tuple.valueHat)
        everyRoundRMSE[t] = (square / testCount) ** 0.5
        everyRoundMAE[t] = abs_count / testCount
        print("round::", t, "everyRoundRMSE:", everyRoundRMSE[t], "everyRoundMAE:", everyRoundMAE[t])
        if everyRoundRMSE[t - 1] - everyRoundRMSE[t] > errorgap:
            if minRMSE > everyRoundRMSE[t]:
                minRMSE = everyRoundRMSE[t]
                minRMSERound = t
            flag_rmse = False
            tr = 0

        if everyRoundMAE[t - 1] - everyRoundMAE[t] > errorgap:
            if minMAE > everyRoundMAE[t]:
                minMAE = everyRoundMAE[t]
                minMAERound = t
            flag_mae = False
            tr = 0
        if flag_rmse and flag_mae:
            tr += 1
            if tr == threshold:
                minRound=t
                break
        flag_rmse = True
        flag_mae = True
    print("**************************************************************************************")
    print("rank:", rank)
    print("testing minRMSE:", minRMSE, " testing minRMSERound:", minRMSERound)
    print("testing minMAE:", minMAE, "testing minMAERound:", minMAERound)
    print("minRound:",minRound)
    return everyRoundRMSE, everyRoundMAE, minRMSERound, minMAERound, minRound,minRMSE,minMAE
