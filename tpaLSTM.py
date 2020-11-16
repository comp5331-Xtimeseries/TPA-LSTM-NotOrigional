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
from util import *;

class TPALSTM(nn.Module):

    def __init__(self, input_size, output_horizon, hidden_size, obs_len, n_layers):
        super(TPALSTM, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(hidden_size, hidden_size, n_layers, \
                    bias=True, batch_first=True) # output (batch_size, obs_len, hidden_size)
        self.hidden_size = hidden_size
        self.filter_num = 32
        self.filter_size = 1
        self.output_horizon = output_horizon
        self.attention = TemporalPatternAttention(self.filter_size, \
            self.filter_num, obs_len-1, hidden_size)
        self.linear = nn.Linear(hidden_size, output_horizon)
        self.n_layers = n_layers

    def forward(self, x):
        batch_size, obs_len = x.size()
        x = x.view(batch_size, obs_len, 1)
        xconcat = self.relu(self.hidden(x))
        # x = xconcat[:, :obs_len, :]
        # xf = xconcat[:, obs_len:, :]
        H = torch.zeros(batch_size, obs_len-1, self.hidden_size)
        ht = torch.zeros(self.n_layers, batch_size, self.hidden_size)
        ct = ht.clone()
        for t in range(obs_len):
            xt = xconcat[:, t, :].view(batch_size, 1, -1)
            out, (ht, ct) = self.lstm(xt, (ht, ct))
            htt = ht.permute(1, 0, 2)
            htt = htt[:, -1, :]
            if t != obs_len - 1:
                H[:, t, :] = htt
        H = self.relu(H)

        # reshape hidden states H
        H = H.view(-1, 1, obs_len-1, self.hidden_size)
        new_ht = self.attention(H, htt)
        ypred = self.linear(new_ht)
        return ypred

class TemporalPatternAttention(nn.Module):

    def __init__(self, filter_size, filter_num, attn_len, attn_size):
        super(TemporalPatternAttention, self).__init__()
        self.filter_size = filter_size
        self.filter_num = filter_num
        self.feat_size = attn_size - self.filter_size + 1
        self.conv = nn.Conv2d(1, filter_num, (attn_len, filter_size))
        self.linear1 = nn.Linear(attn_size, filter_num)
        self.linear2 = nn.Linear(attn_size + self.filter_num, attn_size)
        self.relu = nn.ReLU()

    def forward(self, H, ht):
        _, channels, _, attn_size = H.size()
        new_ht = ht.view(-1, 1, attn_size)
        w = self.linear1(new_ht) # batch_size, 1, filter_num
        conv_vecs = self.conv(H)

        conv_vecs = conv_vecs.view(-1, self.feat_size, self.filter_num)
        conv_vecs = self.relu(conv_vecs)

        # score function
        w = w.expand(-1, self.feat_size, self.filter_num)
        s = torch.mul(conv_vecs, w).sum(dim=2)
        alpha = torch.sigmoid(s)
        new_alpha = alpha.view(-1, self.feat_size, 1).expand(-1, self.feat_size, self.filter_num)
        v = torch.mul(new_alpha, conv_vecs).sum(dim=1).view(-1, self.filter_num)

        concat = torch.cat([ht, v], dim=1)
        new_ht = self.linear2(concat)
        return new_ht
