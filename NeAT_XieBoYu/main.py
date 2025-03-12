import os
import time
import numpy as np
from dotmap import DotMap

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.parameter import Parameter

from model.NeAT import NeAT
from model.metrics import *
from utils.load_data import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(0)

def train(cfg, train_data, test_idxs, test_vals, size):
    dataloader = DataLoader(train_data, batch_size=cfg.batch_size)

    # create the model
    model = NeAT(cfg, size).to(device)
    loss_fn = nn.BCELoss(reduction='mean')
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    m = nn.Sigmoid()

    min_rmse = 1000.0
    min_mae = 1000.0
    RMSE_list = []
    MAE_list = []
    epoch_rmse = 0
    epoch_mae = 0
    total_epoch = 0
    epochs = 1000
    total_start_time = time.time()
    for epoch in range(epochs):
        # train the model
        start = time.time()
        model.train()
        epoch_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            inputs, targets = batch[0], batch[1]
            outputs = model(inputs)
            loss = loss_fn(m(outputs), targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        end = time.time()

        # test the model
        model.eval()
        with torch.no_grad():
            pred = m(model(test_idxs))
            rmse, mae = eval(pred.data, test_vals)
        RMSE_list.append(rmse)
        MAE_list.append(mae)
        print(f"{epoch}, RMSE: {rmse}, MAE: {mae}, time: {end - start}")

        if rmse < min_rmse:
            min_rmse = rmse
            epoch_rmse = epoch + 1
            total_end_rmse = time.time()
        if mae < min_mae:
            min_mae = mae
            epoch_mae = epoch + 1
            total_end_mae = time.time()
        if epoch - epoch_rmse > 10 and epoch - epoch_mae > 10:
            total_time = time.time() - total_start_time
            time_rmse = total_end_rmse - total_start_time
            time_mae = total_end_mae - total_start_time
            total_epoch = epoch + 1
            print(f"min RMSE: {min_rmse}")
            print(f"min MAE: {min_mae}")
            print(f"epoch RMSE: {epoch_rmse}")
            print(f"epoch MAE: {epoch_mae}")
            print(f"time RMSE: {time_rmse}")
            print(f"time MAE: {time_mae}")
            print(f"total time: {total_time}")
            break

    return RMSE_list, MAE_list, epoch_rmse, epoch_mae, total_epoch, time_rmse, time_mae, total_time


def main(cfg, run_times, dataset, val_or_test, idx):
    print(f"rank: {cfg.rank}")
    print(f"layer_dims: {cfg.layer_dims}")
    print(f"learning_rate: {cfg.lr}")
    print(f"weight_decay: {cfg.wd}")
    print(f"dropout: {cfg.dropout}")
    print(f"dropout2: {cfg.dropout2}")
    print(f"batch_size: {cfg.batch_size}")
    # 保存路径
    path = (f"./result/{dataset}_rank={cfg.rank}_layer_dims={cfg.layer_dims}_lr={cfg.lr}_wd={cfg.wd}"
            f"_dropout={cfg.dropout}_{cfg.dropout2}_batch_size={cfg.batch_size}_{idx}.txt")

    # 读取数据集
    train_file = f"./dataset/{dataset}/train.txt"
    test_file = f"./dataset/{dataset}/{val_or_test}.txt"
    train_data, test_idxs, test_vals, size = load_data(train_file, test_file, device)

    # 训练
    all_RMSE = []
    all_MAE = []
    all_epoch_RMSE = []
    all_epoch_MAE = []
    all_total_epoch = []
    all_time_RMSE = []
    all_time_MAE = []
    all_total_time = []
    for t in range(run_times):
        print(f"\n第{t+1}次训练")
        RMSE_list, MAE_list, epoch_rmse, epoch_mae, total_epoch, time_rmse, time_mae, total_time = train(
            cfg, train_data, test_idxs, test_vals, size)
        all_RMSE.append(RMSE_list[epoch_rmse-1])
        all_MAE.append(MAE_list[epoch_mae-1])
        all_epoch_RMSE.append(epoch_rmse)
        all_epoch_MAE.append(epoch_mae)
        all_total_epoch.append(total_epoch)
        all_time_RMSE.append(time_rmse)
        all_time_MAE.append(time_mae)
        all_total_time.append(total_time)
        with open(path, "a", encoding='utf-8') as file:
            file.write(f"第{t+1}次训练\n")
            file.write("RMSE, MAE\n")
            for i in range(total_epoch):
                file.write(f"{RMSE_list[i]}, {MAE_list[i]}\n")
            file.write(f"min RMSE, {RMSE_list[epoch_rmse-1]}\n"
                       f"min MAE, {MAE_list[epoch_mae-1]}\n"
                       f"epoch RMSE, {epoch_rmse}\n"
                       f"epoch MAE, {epoch_mae}\n"
                       f"total epoch, {total_epoch}\n"
                       f"time RMSE, {time_rmse}\n"
                       f"time MAE, {time_mae}\n"
                       f"total time, {total_time}\n\n")

    avg_RMSE = np.mean(all_RMSE)
    avg_MAE = np.mean(all_MAE)
    avg_epoch_RMSE = np.mean(all_epoch_RMSE)
    avg_epoch_MAE = np.mean(all_epoch_MAE)
    avg_total_epoch = np.mean(all_total_epoch)
    avg_time_RMSE = np.mean(all_time_RMSE)
    avg_time_MAE = np.mean(all_time_MAE)
    avg_total_time = np.mean(all_total_time)
    std_RMSE = np.std(all_RMSE)
    std_MAE = np.std(all_MAE)
    std_epoch_RMSE = np.std(all_epoch_RMSE)
    std_epoch_MAE = np.std(all_epoch_MAE)
    std_total_epoch = np.std(all_total_epoch)
    std_time_RMSE = np.std(all_time_RMSE)
    std_time_MAE = np.std(all_time_MAE)
    std_total_time = np.std(all_total_time)
    with open(path, "a", encoding='utf-8') as file:
        file.write(f"avg_RMSE, {avg_RMSE}, std_RMSE, {std_RMSE}\n"
                   f"avg_MAE, {avg_MAE}, std_MAE, {std_MAE}\n"
                   f"avg_epoch_RMSE, {avg_epoch_RMSE}, std_epoch_RMSE, {std_epoch_RMSE}\n"
                   f"avg_epoch_MAE, {avg_epoch_MAE}, std_epoch_MAE, {std_epoch_MAE}\n"
                   f"avg_total_epoch, {avg_total_epoch}, std_total_epoch, {std_total_epoch}\n"
                   f"avg_time_RMSE, {avg_time_RMSE}, std_time_RMSE, {std_time_RMSE}\n"
                   f"avg_time_MAE, {avg_time_MAE}, std_time_MAE, {std_time_MAE}\n"
                   f"avg_total_time, {avg_total_time}, std_total_time, {std_total_time}\n\n")


if __name__ == '__main__':
    for idx in [0]:
        for dataset in ['iawe_1']:
            for rank in [32]:
                for lr in [1e-3]:
                    for wd in [1e-4]:
                        for dropout, dropout2 in [[0.8, 0.2]]:
                            cfg = DotMap()
                            cfg.device = device
                            # 超参数
                            cfg.rank = rank
                            cfg.layer_dims = [3, 32, 1]
                            cfg.depth = len(cfg.layer_dims)
                            cfg.lr = lr
                            cfg.wd = wd
                            cfg.dropout = dropout
                            cfg.dropout2 = dropout2
                            cfg.batch_size = 1024

                            run_times = 5 # 训练次数
                            main(cfg, run_times, dataset, "test", idx)
