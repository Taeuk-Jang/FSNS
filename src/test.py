#!/usr/bin/env python
# coding: utf-8
import sys
import os
import numpy as np
from utils import *
import eval_utils
import models
from models import Feature_extractor, Feature_predictor, SVM

sys.path.append("../")
import dataloader
# import pandas as pd
import torch
import torch.utils.data as data

sys.path.append("../")
from MulticoreTSNE import MulticoreTSNE as TSNE

from aif360.datasets import AdultDataset, GermanDataset, BankDataset, CompasDataset, BinaryLabelDataset, CelebADataset
from aif360.metrics import ClassificationMetric
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers, initializers, regularizers, metrics

import tensorflow as tf
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import scale, StandardScaler, MaxAbsScaler
from sklearn.metrics import accuracy_score

import argparse
from plot import plot
from keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import copy
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import sklearn.metrics as sklm
import logging
import csv

from tensorboardX import SummaryWriter


from math import isnan
from datetime import datetime
import itertools

datetime_dir = datetime.today().strftime("%Y-%m-%d")
print(datetime_dir)


parser = argparse.ArgumentParser()
# parser.add_argument('--dropouts', nargs='+', type=float, default=[0.1,0.1,0.1])
parser.add_argument('--dataset', type=str, default='compas')
parser.add_argument('--senstive_feature', type=int, default=1)

parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--upsample', type=bool, default=False)

parser.add_argument('--lr_h', type=float, default=1e-3)
parser.add_argument('--lr_c', type=float, default=1e-5)
parser.add_argument('--lr_p', type=float, default=1e-3)

parser.add_argument('--eta', type=float, default=1)
parser.add_argument('--lamda', type=float, default=1)
parser.add_argument('--alpha', type=float, default=0.2)
parser.add_argument('--mu', type=float, default=0.6)

parser.add_argument('--latent', type=int, default=64)
parser.add_argument('--hidden', type=int, default=64)

parser.add_argument('--repeat', type=int, default=5)
parser.add_argument('--gpu_id', type=str, default='0')
parser.add_argument('--workers', type=int, default=16)
args = parser.parse_args()

data_name = args.dataset
protected_attribute_used = args.senstive_feature
device = 'cuda:'+args.gpu_id
bs = args.batch_size
workers = args.workers

if data_name == "adult":
    dataset_orig = AdultDataset()
    # #dataset_orig = load_preproc_data_adult()
    if protected_attribute_used == 1:
        privileged_groups = [{'sex': 1}]
        unprivileged_groups = [{'sex': 0}]
        sens_attr = 'sex'
    else:
        privileged_groups = [{'race': 1}]
        unprivileged_groups = [{'race': 0}]
        sens_attr = 'race'

elif data_name == "german":
    dataset_orig = GermanDataset()
    # #dataset_orig.labels = #dataset_orig.labels-1
    if protected_attribute_used == 1:
        privileged_groups = [{'sex': 1}]
        unprivileged_groups = [{'sex': 0}]
        sens_attr = 'sex'
    else:
        privileged_groups = [{'age': 1}]
        unprivileged_groups = [{'age': 0}]
        sens_attr = 'age'

elif data_name == "compas":
    dataset_orig = CompasDataset()
    # #dataset_orig = load_preproc_data_compas()
    if protected_attribute_used == 1:
        privileged_groups = [{'sex': 1}]
        unprivileged_groups = [{'sex': 0}]
        sens_attr = 'sex'
    else:
        privileged_groups = [{'race': 1}]
        unprivileged_groups = [{'race': 0}]
        sens_attr = 'race'

elif data_name == "bank":
    dataset_orig = BankDataset()
    if protected_attribute_used == 1:
        privileged_groups = [{'age': 1}]
        unprivileged_groups = [{'age': 0}]
        sens_attr = 'age'
    else:
        privileged_groups = [{'age': 1}]
        unprivileged_groups = [{'age': 0}]
        sens_attr = 'age'
        
elif data_name == "meps":
    dataset_orig = MEPSDataset19()
    privileged_groups = [{'RACE': 1}]
    unprivileged_groups = [{'RACE': 0}]
    sens_attr = 'RACE'
    
sens_idx = dataset_orig.feature_names.index(sens_attr)
    
min_max_scaler = MaxAbsScaler()
    
if args.upsample:
    data_train, data_valid, data_test = upsample_all_equal(dataset_orig, np.array([sens_idx]))
    
else:
    data_train, data_vt = dataset_orig.split([0.7], shuffle=True)
    data_valid, data_test = data_vt.split([0.5], shuffle=True)

d_train = Dataset(data_train, sens_idx)
d_valid = Dataset(data_valid, sens_idx)
d_test = Dataset(data_test, sens_idx)

trainloader = torch.utils.data.DataLoader(
    d_train,
    batch_size=bs,
    shuffle=True,
    num_workers=workers)

validloader = torch.utils.data.DataLoader(
    d_valid,
    batch_size=bs,
    shuffle=True,
    num_workers=workers)
testloader = torch.utils.data.DataLoader(
    d_test,
    batch_size=bs,
    shuffle=True,
    num_workers=workers)


epochs = 200

input_size = data_train.features.shape[1]-1
num_sens = len(np.unique(data_train.features[:, sens_idx]))
latent_size = args.latent
hidden_size = args.hidden

FIG_EPOCH = 100
VIEW_FIG = False
SAVE_MODEL = False
TEST_EPOCH = 5
VIEW_EVAL = False


eta = args.eta
lamda = args.lamda
alpha = args.alpha
mu = args.mu
epoch = 0 

tpr_overall_hist, tpr_priv_hist, tpr_unpriv_hist, acc_diff_hist, \
eq_opp_hist, fpr_overall_hist, fpr_priv_hist, fpr_unpriv_hist, \
fpr_diff_hist, acc_overall_hist, acc_priv_hist, acc_unpriv_hist, \
balanced_hist, dis_impact_hist, theil_hist, stat_hist\
= [], [], [], [], \
[], [], [], [], \
[], [], [], [], \
[], [], [], []

for repeat in range(args.repeat):
        tpr_overall_value, tpr_priv_value, tpr_unpriv_value, acc_diff_value, \
        eq_opp_value, fpr_overall_value, fpr_priv_value, fpr_unpriv_value, \
        fpr_diff_value, acc_overall_value, acc_priv_value, acc_unpriv_value, \
        balanced_value, dis_impact_value, theil_value, stat_value\
        = [], [], [], [], \
        [], [], [], [], \
        [], [], [], [], \
        [], [], [], []

        config = 'alpha:{}, mu:{}, lr_h:{}, lr_c:{}, lr_p:{}'\
                               .format(alpha, mu, args.lr_h, args.lr_c, args.lr_p)

        tsne = TSNE(n_jobs=16)

        H = Feature_extractor(input_size, hidden_size, latent_size).double().to(device) # input_size -> 1024
        P = Feature_predictor(latent_size, hidden_size, 2).double().to(device)
        C = SVM(latent_size, 2).double().to(device) # 1024 -> 2   

        H.apply(models.init_weights)
        P.apply(models.init_weights)
        C.apply(models.init_weights)
        
        #modify the file path to test with different saved model.

        H.load_state_dict(torch.load('../model/{}/extractor.pth'.format(data_name)))
        P.load_state_dict(torch.load('../model/{}/predictor.pth'.format(data_name)))
        C.load_state_dict(torch.load('../model/{}/classifier.pth'.format(data_name)))

#         latent = H(x)
#         vis_utils.vis(latent, a, y)

        VIEW_EVAL = eval_utils.evaluate(args, repeat, epoch, data_valid, validloader, H, P, C, sens_idx, num_sens, privileged_groups, unprivileged_groups, device, False)

        tpr_overall, tpr_priv, tpr_unpriv, eq_opp, \
        fpr_overall, fpr_priv, fpr_unpriv, fpr_diff, \
        acc_overall, acc_priv, acc_unpriv, acc_diff, \
        bal_acc, dis_imp, theil, stat = eval_utils.evaluate(args, repeat, epoch, data_test, testloader, H, P, C, sens_idx, num_sens, privileged_groups, unprivileged_groups, device, True)

        tpr_overall_value.append(tpr_overall)
        tpr_priv_value.append(tpr_priv)
        tpr_unpriv_value.append(tpr_unpriv)
        eq_opp_value.append(eq_opp)

        fpr_overall_value.append(fpr_overall)
        fpr_priv_value.append(fpr_priv)
        fpr_unpriv_value.append(fpr_unpriv)
        fpr_diff_value.append(fpr_diff)

        acc_overall_value.append( acc_overall)
        acc_priv_value.append(acc_priv)
        acc_unpriv_value.append(acc_unpriv)
        acc_diff_value.append(acc_diff)

        balanced_value.append(bal_acc)
        dis_impact_value.append(dis_imp)
        theil_value.append(theil)
        stat_value.append(stat)

#         vis_utils.vis(latent_val, a_val, y_val)







