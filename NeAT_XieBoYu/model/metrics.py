
import torch
import torchmetrics.functional as tmf

def eval(pred, true):
    rmse = torch.sqrt(torch.mean(torch.square(pred - true))).item()
    mae = torch.mean(torch.abs(pred - true)).item()
    return rmse, mae

def rmse(pred, true):
    ''' Calculate Root Mean Square Error (RMSE)'''
    return torch.sqrt(torch.mean(torch.square(pred - true)))

def nre(pred, true):
    ''' Calculate Normalized reconstruction Error '''
    diff1 = torch.sqrt(torch.sum(torch.square(pred - true)))
    diff2 = torch.sqrt(torch.sum(torch.square(true)))
    return diff1/diff2

def eval_(pred, true, q=0.5, task='binary'):
    ''' Calculate classification metrics'''
    acc = tmf.accuracy(pred>q, true.long(), task=task)
    recall = tmf.recall(pred>q, true.long(), task=task)
    prec = tmf.precision(pred>q, true.long(), task =task)
    f1 = tmf.f1_score(pred, true.long(), task=task)
    auc = tmf.auroc(pred, true.long(), task=task)
        
    return {"acc" :acc.item(), "recall":recall.item(),
            "prec":prec.item(),"f1":f1.item(), "auc":auc.item()}
        