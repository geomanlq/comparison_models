import torch as t
import Utils.TimeLogger as logger
from Utils.TimeLogger import log
from Params import args
from Model import Model, RandomMaskSubgraphs, LocalGraph, GTLayer
from DataHandler import DataHandler
import pickle
from Utils.Utils import *
from Utils.Utils import contrast
import os
import torch.nn as nn
import time
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


class Coach:
    def __init__(self, handler):
        self.handler = handler

        print('USER', args.user, 'ITEM', args.item)
        print('NUM OF INTERACTIONS', self.handler.trnLoader.dataset.__len__())
        self.metrics = dict()
        mets = ['Loss', 'preLoss', 'Recall', 'NDCG']
        for met in mets:
            self.metrics['Train' + met] = list()
            self.metrics['Test' + met] = list()

    def makePrint(self, name, ep, reses, save):
        ret = 'Epoch %d/%d, %s: ' % (ep, args.epoch, name)
        for metric in reses:
            val = reses[metric]
            ret += '%s = %.7f, ' % (metric, val)
            tem = name + metric
            if save and tem in self.metrics:
                self.metrics[tem].append(val)
        ret = ret[:-2] + '  '
        return ret

    def run(self):
        best_RMSE = 100
        best_MAE = 100
        self.prepareModel()
        log('Model Prepared')
        if args.load_model != None:
            self.loadModel()
            stloc = len(self.metrics['TrainLoss']) * args.tstEpoch - (args.tstEpoch - 1)
        else:
            stloc = 0
            log('Model Initialized')
        bestRes = None
        # result = []
        t1 = time.time()
        for ep in range(stloc, args.epoch):
            tstFlag = (ep % args.tstEpoch == 0)
            reses = self.trainEpoch()
            log(self.makePrint('Train', ep, reses, tstFlag))
            # if tstFlag:
            val_reses = self.valEpoch()
            reses = self.testEpoch()
            log(self.makePrint('Val', ep, val_reses, tstFlag))

            log(self.makePrint('Test', ep, reses, tstFlag))
            t2 = time.time()
            print("untiltime", t2 - t1)
            # self.saveHistory()
            if (reses['RMSE'] < best_RMSE):
                best_RMSE = reses['RMSE']
            if (reses['MAE'] < best_MAE):
                best_MAE = reses['MAE']
            # result.append(reses)
            # bestRes = reses if bestRes is None or reses['Recall'] > bestRes['Recall'] else bestRes
            print()
        print("RMSE,MAE", best_RMSE, best_MAE)

        # reses = self.testEpoch()
        # result.append(reses)
        # torch.save(result, "Saeg_result.pkl")
        # log(self.makePrint('Test', args.epoch, reses, True))
        # log(self.makePrint('Best Result', args.epoch, bestRes, True))
        # self.saveHistory()

    def prepareModel(self):
        self.gtLayer = GTLayer().cuda()
        self.model = Model(self.gtLayer).cuda()
        self.opt = t.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=0)
        self.masker = RandomMaskSubgraphs(args.user, args.item)
        self.sampler = LocalGraph(self.gtLayer)

    def huber_loss(self,pos_scores, neg_scores, delta=1.0):
        diff = pos_scores - neg_scores
        abs_diff = t.abs(diff)
        quadratic = t.minimum(abs_diff, t.tensor(delta).cuda())
        linear = abs_diff - quadratic
        loss = 0.5 * quadratic ** 2 + delta * linear
        return loss.mean()

    def trainEpoch(self):
        trnLoader = self.handler.trnLoader
        trnLoader.dataset.negSampling()
        epLoss, epPreLoss = 0, 0
        # epLoss = 0

        steps = trnLoader.dataset.__len__() // args.batch
        self.handler.preSelect_anchor_set()
        for i, tem in enumerate(trnLoader):
            if i % args.fixSteps == 0:
                att_edge, add_adj = self.sampler(self.handler.torchBiAdj, self.model.getEgoEmbeds(),
                                                 self.handler)
                encoderAdj, decoderAdj, sub, cmp = self.masker(add_adj, att_edge)
            # ancs, poss, negs = tem
            ancs, poss, negs, truth = tem

            ancs = ancs.long().cuda()
            poss = poss.long().cuda()
            negs = negs.long().cuda()
            truth = truth.float().cuda()
            usrEmbeds, itmEmbeds, cList, subLst = self.model(self.handler, False, sub, cmp,  encoderAdj,decoderAdj)
            ancEmbeds = usrEmbeds[ancs]
            posEmbeds = itmEmbeds[poss]
            negEmbeds = itmEmbeds[negs]
            pos_scores = (ancEmbeds * posEmbeds).sum(-1)
            neg_scores = (ancEmbeds * negEmbeds).sum(-1)

            actual_scores = truth
            rmse_loss = t.sqrt(((pos_scores - actual_scores) ** 2).mean())
            # huberloss = self.huber_loss(pos_scores, neg_scores)

            usrEmbeds2 = subLst[:args.user]
            itmEmbeds2 = subLst[args.user:]
            ancEmbeds2 = usrEmbeds2[ancs]
            posEmbeds2 = itmEmbeds2[poss]

            # bprLoss = (-t.sum(ancEmbeds * posEmbeds, dim=-1)).mean()
            #
            scoreDiff = pairPredict(ancEmbeds2, posEmbeds2, negEmbeds)
            bprLoss2 = - (scoreDiff).sigmoid().log().sum() / args.batch

            regLoss = calcRegLoss(self.model) * args.reg

            contrastLoss = (contrast(ancs, usrEmbeds) + contrast(poss, itmEmbeds)) * args.ssl_reg + contrast(
                ancs,
                usrEmbeds,
                itmEmbeds) + args.ctra*contrastNCE(ancs, subLst, cList)
            # loss = bprLoss + regLoss + contrastLoss + args.b2*bprLoss2
            loss = rmse_loss + regLoss + contrastLoss + args.b2*bprLoss2
            # loss = huberloss+ regLoss

            epLoss += loss.item()
            # epPreLoss += bprLoss.item()
            # epPreLoss += rmse_loss.item()

            self.opt.zero_grad()
            # rmse_loss.backward()
            loss.backward()

            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=20, norm_type=2)
            self.opt.step()
            log('Step %d/%d: loss = %.3f, regLoss = %.3f, clLoss = %.3f        ' % (
                i, steps, loss, regLoss, contrastLoss), save=False, oneline=True)
            # log('Step %d/%d: loss = %.4f        ' % (
            #     i, steps, loss), save=False, oneline=True)
            # log('Step %d/%d: loss = %.3f, regLoss = %.3f, clLoss = %.3f        ' % (
            #     i, steps, loss, regLoss, contrastLoss), save=False, oneline=True)
        ret = dict()
        ret['Loss'] = epLoss / steps
        ret['preLoss'] = epPreLoss / steps
        return ret

    def testEpoch(self):
        tstLoader = self.handler.tstLoader
        # epLoss, epRecall, epNdcg = [0] * 3
        total_squared_error = 0.0
        total_absolute_error = 0.0
        total_count = 0

        i = 0
        num = tstLoader.dataset.__len__()
        steps = num // args.tstBat

        for usr, trnMask in tstLoader:
            i += 1
            usr = usr.long().cuda()
            trnMask = trnMask.cuda()
            usrEmbeds, itmEmbeds, _, _ = self.model(self.handler, True, self.handler.torchBiAdj, self.handler.torchBiAdj,
                                                          self.handler.torchBiAdj)

            allPreds = t.mm(usrEmbeds[usr], t.transpose(itmEmbeds, 1, 0)) * (1 - trnMask) - trnMask * 1e8

            allPreds = t.sigmoid(allPreds)

            truth = self.handler.tstLoader.dataset.gettruth_for_users(usr)
            # 计算 RMSE 和 MAE
            mask = (truth != 0)
            truth = t.tensor(truth, dtype=t.float32).cuda()
            allPreds = allPreds[:, :truth.shape[1]]

            # print("-pred", allPreds)
            # print("-truth", truth)
            squared_error = t.sum((allPreds[mask] - truth[mask]) ** 2)
            absolute_error = t.sum(t.abs(allPreds[mask] - truth[mask]))
            mask_tensor = t.tensor(mask)
            count = t.sum(mask_tensor.float())

            total_squared_error += squared_error.item()
            total_absolute_error += absolute_error.item()
            total_count += count.item()

            # 输出当前步骤的 RMSE 和 MAE
            rmse_step = (squared_error / count) ** 0.5 if count != 0 else 0.0
            mae_step = absolute_error / count if count != 0 else 0.0
            log('Steps %d/%d: test-RMSE = %.7f, test-MAE = %.7f' % (i, steps, rmse_step, mae_step), save=False,
                oneline=True)

            # _, topLocs = t.topk(allPreds, args.topk)
            # recall, ndcg = self.calcRes(topLocs.cpu().numpy(), self.handler.tstLoader.dataset.tstLocs, usr)
            # epRecall += recall
            # epNdcg += ndcg
            # log('Steps %d/%d: recall = %.2f, ndcg = %.2f          ' % (i, steps, recall, ndcg), save=False,
            #     oneline=True)
        ret = dict()
        ret['RMSE'] = (total_squared_error / total_count) ** 0.5 if total_count != 0 else 0.0
        ret['MAE'] = total_absolute_error / total_count if total_count != 0 else 0.0

        # ret['Recall'] = epRecall / num
        # ret['NDCG'] = epNdcg / num
        return ret

    def valEpoch(self):
        valLoader = self.handler.valLoader
        total_squared_error = 0.0
        total_absolute_error = 0.0
        total_count = 0
        i = 0
        num = valLoader.dataset.__len__()
        steps = num // args.tstBat
        for usr, trnMask in valLoader:
            i += 1
            usr = usr.long().cuda()
            trnMask = trnMask.cuda()
            usrEmbeds, itmEmbeds, _, _ = self.model(self.handler, True, self.handler.torchBiAdj,self.handler.torchBiAdj,
                                                    self.handler.torchBiAdj)
            allPreds = t.mm(usrEmbeds[usr], t.transpose(itmEmbeds, 1, 0)) * (1 - trnMask) - trnMask * 1e8
            allPreds = t.sigmoid(allPreds)

            truth = self.handler.valLoader.dataset.gettruth_for_users(usr)
            allPreds = allPreds[:, :truth.shape[1]]

            # 计算 RMSE 和 MAE
            mask = (truth != 0)
            truth = t.tensor(truth, dtype=t.float32).cuda()
            # print(allPreds.shape)
            # print(truth.shape)
            # print(mask.shape)
            # print("pred",allPreds)
            # print("truth",truth)
            squared_error = t.sum((allPreds[mask] - truth[mask]) ** 2)
            absolute_error = t.sum(t.abs(allPreds[mask] - truth[mask]))
            mask_tensor = t.tensor(mask)
            count = t.sum(mask_tensor.float())

            total_squared_error += squared_error.item()
            total_absolute_error += absolute_error.item()
            total_count += count.item()

            # 输出当前步骤的 RMSE 和 MAE
            rmse_step = (squared_error / count) ** 0.5 if count != 0 else 0.0
            mae_step = absolute_error / count if count != 0 else 0.0
            log('Steps %d/%d: valRMSE = %.7f, valMAE = %.7f' % (i, steps, rmse_step, mae_step), save=False,
                oneline=True)
        ret = dict()
        ret['RMSE'] = (total_squared_error / total_count) ** 0.5 if total_count != 0 else 0.0
        ret['MAE'] = total_absolute_error / total_count if total_count != 0 else 0.0

        return ret

    def calcRes(self, topLocs, tstLocs, batIds):
        assert topLocs.shape[0] == len(batIds)
        allRecall = allNdcg = 0
        for i in range(len(batIds)):
            temTopLocs = list(topLocs[i])
            temTstLocs = tstLocs[batIds[i]]
            tstNum = len(temTstLocs)
            maxDcg = np.sum([np.reciprocal(np.log2(loc + 2)) for loc in range(min(tstNum, args.topk))])
            recall = dcg = 0
            for val in temTstLocs:
                if val in temTopLocs:
                    recall += 1
                    dcg += np.reciprocal(np.log2(temTopLocs.index(val) + 2))
            recall = recall / tstNum
            ndcg = dcg / maxDcg
            allRecall += recall
            allNdcg += ndcg
        return allRecall, allNdcg

    def saveHistory(self):
        if args.epoch == 0:
            return
        with open('./History/' + args.save_path + '.his', 'wb') as fs:
            pickle.dump(self.metrics, fs)

        content = {
            'model': self.model,
        }
        t.save(content, './Models/' + args.save_path + '.mod')
        log('Model Saved: %s' % args.save_path)

    def loadModel(self):
        ckp = t.load('./Models/' + args.load_model + '.mod')
        self.model = ckp['model']
        self.opt = t.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=0)

        with open('./History/' + args.load_model + '.his', 'rb') as fs:
            self.metrics = pickle.load(fs)
        log('Model Loaded')


if __name__ == '__main__':
    logger.saveDefault = True

    log('Start')
    if t.cuda.is_available():
        print("using cuda")
    handler = DataHandler()
    handler.LoadData()
    log('Load Data')

    coach = Coach(handler)
    coach.run()
