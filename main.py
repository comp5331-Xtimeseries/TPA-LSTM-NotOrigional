#!/usr/bin/python 3.7
#-*-coding:utf-8-*-

'''
Pytorch Implementation of TPA-LSTM
Paper link: https://arxiv.org/pdf/1809.04206v2.pdf
Author: Jing Wang (jingw2@foxmail.com)
Date: 04/10/2020
'''

import torch
from torch import nn
import torch.nn.functional as F
import argparse
from progressbar import *
from torch.optim import Adam
import util
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import os
import pandas as pd
from datetime import date
from util import *
from tpaLSTM import TPALSTM
import tpaLSTM
import Datasets

import random
import matplotlib.pyplot as plt
# import tensorflow as tf
def evaluate(data, X, Y, model, evaluateL2, evaluateL1, batch_size, args, yscaler):
    model.eval();

    total_loss = 0;
    total_loss_l1 = 0;
    n_samples = 0;
    predict = None;
    test = None;

    seq_len = args.seq_len
    obs_len = args.num_obs_to_train

    for step in range(args.step_per_epoch):
        Xeva, yeva, Xf, yf, batch = util.batch_generator(X, Y, obs_len, seq_len, args.batch_size)
        Xeva = torch.from_numpy(Xeva).float()
        yeva = torch.from_numpy(yeva).float()
        Xf = torch.from_numpy(Xf).float()
        yf = torch.from_numpy(yf).float()

        for i in range(len(Xeva[0][0])):

            yeva = Xeva[:,:,i]
            yf =  Xf[:,:,i]

            ypred = model(yeva)

            scale = data.scale[batch]
            scale = scale.view([scale.size(0),1])

            ypred = ypred*scale
            yf = yf*scale

            ypred = ypred.data.numpy()
            if yscaler is not None:
                ypred = yscaler.inverse_transform(ypred)
            ypred = ypred.ravel()

            yfs  = yf.shape
            ypred = ypred.ravel().reshape(yfs[0], yfs[1])

            ypred = torch.Tensor(ypred)
            yf = torch.Tensor(yf)

            if torch.isnan(yf).any():
                continue

            if predict is None:
                predict = ypred;
                test = yf;
            else:
                predict = torch.cat((predict,ypred));
                test = torch.cat((test, yf));

            total_loss += evaluateL2(ypred, yf ).item()
            total_loss_l1 += evaluateL1(ypred, yf).item()

            n_samples += (yf.size(0))
            # n_samples += (yf.size(0) * data.m)


    rse = math.sqrt(total_loss / n_samples)/data.rse
    rae = (total_loss_l1/n_samples)/data.rae

    predict = predict.data.cpu().numpy();
    Ytest = test.data.cpu().numpy();
    sigma_p = (predict).std(axis = 0);
    sigma_g = (Ytest).std(axis = 0);
    mean_p = predict.mean(axis = 0)
    mean_g = Ytest.mean(axis = 0)
    index = (sigma_g!=0);
    correlation = ((predict - mean_p) * (Ytest - mean_g)).mean(axis = 0)/(sigma_p * sigma_g);
    correlation = (correlation[index]).mean();

    return rse, rae, correlation;

def train(Data,args):
    '''
    Args:
    - X (array like): shape (num_samples, num_features, num_periods)
    - y (array like): shape (num_samples, num_periods)
    - epoches (int): number of epoches to run
    - step_per_epoch (int): steps per epoch to run
    - seq_len (int): output horizon
    - likelihood (str): what type of likelihood to use, default is gaussian
    - num_skus_to_show (int): how many skus to show in test phase
    - num_results_to_sample (int): how many samples in test phase as prediction
    '''

    evaluateL2 = nn.MSELoss(size_average=False);
    evaluateL1 = nn.L1Loss(size_average=False)

    if args.L1Loss:
        criterion = nn.L1Loss(size_average=False);
    else:
        criterion = nn.MSELoss(size_average=False);

    yscaler = None
    if args.standard_scaler:
        yscaler = util.StandardScaler()
    elif args.log_scaler:
        yscaler = util.LogScaler()
    elif args.mean_scaler:
        yscaler = util.MeanScaler()
    elif args.max_scaler:
        yscaler = util.MaxScaler()


    # model = TPALSTM(1, args.seq_len, args.hidden_size, args.num_obs_to_train, args.n_layers)

    modelPath = "/home/isabella/Documents/5331/tpaLSTM/model/electricity.pt"

    with open(modelPath, 'rb') as f:
        model = torch.load(f)


    optimizer = Adam(model.parameters(), lr=args.lr)
    random.seed(2)

    # select sku with most top n quantities
    Xtr = np.asarray(Data.train[0].permute(2,0,1))
    ytr = np.asarray(Data.train[1].permute(1,0))
    Xte = np.asarray(Data.test[0].permute(2,0,1))
    yte = np.asarray(Data.test[1].permute(1,0))
    Xeva = np.asarray(Data.valid[0].permute(2,0,1))
    yeva = np.asarray(Data.valid[1].permute(1,0))

    # print("\nRearranged Data")
    # print("Xtr.size", Xtr.shape)
    # print("ytr.size", ytr.shape)
    # print("Xte.size", Xte.shape)
    # print("yte.size", yte.shape)
    # print("Xeva.size", Xeva.shape)
    # print("yeva.size", yeva.shape)

    num_ts, num_periods, num_features = Xtr.shape

    if yscaler is not None:
        ytr = yscaler.fit_transform(ytr)

    # training
    seq_len = args.seq_len
    obs_len = args.num_obs_to_train
    progress = ProgressBar()
    best_val = np.inf
    total_loss = 0;
    n_samples = 0
    losses = []
    for epoch in progress(range(args.num_epoches)):
        epoch_start_time = time.time()
        model.train()
        total_loss = 0;
        n_samples = 0;
        # print("\n\nData.get_batches")
        # for X,Y in Data.get_batches(Data.train[0], Data.train[1], args.batch_size, True):
        #     print("X.shape",X.shape)
        #     print("Y.shape", Y.shape)

        for step in range(args.step_per_epoch):
            print(step)
            Xtrain, ytrain, Xf, yf, batch = util.batch_generator(Xtr, ytr, obs_len, seq_len, args.batch_size)
            Xtrain = torch.from_numpy(Xtrain).float()
            ytrain = torch.from_numpy(ytrain).float()
            Xf = torch.from_numpy(Xf).float()
            yf = torch.from_numpy(yf).float()

            for i in range(len(Xeva[0][0])):

                ytrain = Xtrain[:,:,i]
                yf =  Xf[:,:,i]

                ypred = model(ytrain)
                scale = Data.scale[batch]
                scale = scale.view([scale.size(0),1])

                loss = criterion(ypred*scale, yf*scale)

                losses.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item();
                n_samples += (ypred.size(0));

        train_loss = total_loss / n_samples

        val_loss, val_rae, val_corr = evaluate(Data, Xeva,yeva, model, evaluateL2, evaluateL1, args.batch_size, args, yscaler);
        print('| end of epoch {:3d} | time: {:5.2f}s | train_loss {:5.4f} | valid rse {:5.4f} | valid rae {:5.4f} | valid corr  {:5.4f}'.
        format(epoch, (time.time() - epoch_start_time), train_loss, val_loss, val_rae, val_corr))

        # Save the model if the validation loss is the best we've seen so far.
        if val_loss < best_val:
            with open(args.save, 'wb') as f:
                torch.save(model, f)
            best_val = val_loss

        if epoch % 5 == 0:
            test_acc, test_rae, test_corr  = evaluate(Data, Xte, yte, model, evaluateL2, evaluateL1, args.batch_size, args, yscaler);
            print ("test rse {:5.4f} | test rae {:5.4f} | test corr {:5.4f}".
            format(test_acc, test_rae, test_corr))


    # test_acc, test_rae, test_corr  = evaluate(Data, Xte, yte, model, evaluateL2, evaluateL1, args.batch_size, args, yscaler);
    # print ("test rse {:5.4f} | test rae {:5.4f} | test corr {:5.4f}".
    # format(test_acc, test_rae, test_corr))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True,
                        help='location of the data file')
    parser.add_argument('--cuda', type=str, default=True)
    parser.add_argument('--horizon', type=int, default=12)
    parser.add_argument('--window', type=int, default=24 * 7,
                        help='window size')
    parser.add_argument('--normalize', type=int, default=2)
    parser.add_argument('--model', type=str, default='tpaLSTM',
                        help='')
    parser.add_argument('--L1Loss', type=bool, default=False)
    parser.add_argument('--save', type=str,  default='model/model.pt',
                        help='path to save the final model')

    parser.add_argument("--num_epoches", "-e", type=int, default=1000)
    parser.add_argument("--step_per_epoch", "-spe", type=int, default=2)
    parser.add_argument("-lr", type=float, default=1e-3)
    parser.add_argument("--n_layers", "-nl", type=int, default=3)
    parser.add_argument("--hidden_size", "-hs", type=int, default=24)
    parser.add_argument("--seq_len", "-sl", type=int, default=7)
    parser.add_argument("--num_obs_to_train", "-not", type=int, default=1)
    parser.add_argument("--num_results_to_sample", "-nrs", type=int, default=10)
    parser.add_argument("--show_plot", "-sp", action="store_true")
    parser.add_argument("--standard_scaler", "-ss", action="store_true")
    parser.add_argument("--log_scaler", "-ls", action="store_true")
    parser.add_argument("--mean_scaler", "-ms", action="store_true")
    parser.add_argument("--max_scaler", "-max", action="store_true")
    parser.add_argument("--batch_size", "-b", type=int, default=64)

    args = parser.parse_args()

    if args.data=="solar":
      dSet=Datasets.Solar().data
    elif args.data=="exchange_rate":
      dSet=Datasets.ExchangeRate().data
    elif args.data=="electricity":
      dSet=Datasets.Electricity().data
    elif args.data=="traffic":
      dSet=Datasets.Traffic().data

    print("dSet.shape", dSet.shape)
    Data = Data_utility(dSet, 0.6, 0.2, False, args.horizon, args.window, args.normalize);
    # Data = Data_utility(dSet, 0.6, 0.2, args.cuda, args.horizon, args.window, args.normalize);

    print(Data.rse);
    train(Data , args)
