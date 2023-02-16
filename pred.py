#!/usr/bin/env python
# coding: utf-8

#------ Load necessary packages ------#
import sys
import os
import numpy as np
import dataloader
# import pandas as pd
import torch
import torch.utils.data as data
#from common_utils import compute_metrics
sys.path.append("../")

# load datasets
from aif360.datasets import AdultDataset, GermanDataset, BankDataset, CompasDataset, BinaryLabelDataset, CelebADataset

# load metrics
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.metrics import ClassificationMetric
# from aif360.metrics.utils import compute_boolean_conditioning_vector

# load preprocessing algorithm
from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import load_preproc_data_compas
# from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import load_preproc_data_adult
# from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import load_preproc_data_german
# from aif360.algorithms.preprocessing.optim_preproc_helpers.opt_tools import OptTools
# from aif360.algorithms.preprocessing.optim_preproc_helpers.distortion_functions import get_distortion_adult

# load algorithms
from aif360.algorithms.preprocessing import DisparateImpactRemover
from aif360.algorithms.preprocessing.lfr import LFR
from aif360.algorithms.preprocessing.optim_preproc import OptimPreproc
from aif360.algorithms.preprocessing.reweighing import Reweighing
from aif360.algorithms.inprocessing.prejudice_remover import PrejudiceRemover
from aif360.algorithms.inprocessing.adversarial_debiasing import AdversarialDebiasing
from aif360.algorithms.postprocessing.calibrated_eq_odds_postprocessing import CalibratedEqOddsPostprocessing
from aif360.algorithms.postprocessing.reject_option_classification import RejectOptionClassification

import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers, initializers, regularizers, metrics

# load other packages
import tensorflow as tf
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import scale, StandardScaler, MaxAbsScaler
from sklearn.metrics import accuracy_score
#from function_1 import *
#from fair_main import *
#import fair_ablation
import argparse
from plot import plot
from keras.models import load_model

from sklearn.model_selection import train_test_split
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse

import sklearn.metrics as sklm
import logging
import csv

from model import *
from util import *

from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument('--dataset',  default = 'compas', type = str)
parser.add_argument('--device',  default = 'cuda:0', type = str)
parser.add_argument('--bs',  default = 256, type = int)

args = parser.parse_args()

#dataset = {'adult' : AdultDataset(), 'german' : GermanDataset(),'bank': BankDataset(),'compas' : CompasDataset(),'celeb': CelebADataset()}
dataset = {'adult' : AdultDataset(), 'german' : GermanDataset(),'bank': BankDataset(),'compas' : CompasDataset()}
dataset['german'].labels -= 1
sens_attr_dict = {'adult': ['sex', 'race'], 'german' : ['sex', 'age'], 'compas' : ['sex', 'race'], 'bank' : ['age'], 'celeb' : ['gender']}

np.random.seed(0)

data_name = args.dataset
device = args.device
bs = args.bs

sens_attr = sens_attr_dict[data_name][0]
sens_idx = dataset[data_name].feature_names.index(sens_attr)

#bs = 4000
workers = 16
hist = {'loss':[], 'acc':[], 'val_loss':[], 'val_acc':[]}

dataset[data_name].features[:, sens_idx] += 1

data_train, data_vt = dataset[data_name].split([0.7], shuffle=True)
data_valid, data_test = data_vt.split([0.5], shuffle=True)

d_train = Dataset(data_train)
d_valid = Dataset(data_valid)
d_test = Dataset(data_test)


#bs_train, bs_valid, bs_test = len(d_train), len(d_valid), len(d_test)
bs_train, bs_valid, bs_test = bs, bs, bs

trainloader = torch.utils.data.DataLoader(
    d_train,
    batch_size=bs_train,
    shuffle=False,
    num_workers=workers)

validloader = torch.utils.data.DataLoader(
    d_valid,
    batch_size=bs_valid,
    shuffle=False,
    num_workers=workers)

testloader = torch.utils.data.DataLoader(
    d_test,
    batch_size=bs_test,
    shuffle=False,
    num_workers=workers)

input_shape = d_train[0][0].shape[0]

beta_list = [1e2]
lr_list = [1e-5]

alpha_p_list = [1]
alpha_n_list = [1]

epochs  = 200

for beta in beta_list:
    for lr in lr_list:
        for alpha_p in alpha_p_list:
            for alpha_n in alpha_n_list:
                config = '{}/{}_non_pca_new  lr : {:.5f}, beta : {}, alpha : {}'.format(data_name, sens_attr, lr, beta, alpha_n)
                print(beta)
                print(lr)
                print(alpha_p)
                print(alpha_n)
                best_loss = np.inf
                best_acc = 0
                best_epoch = 0

                model = deep_network(input_shape)
                celoss = nn.functional.cross_entropy
                mseloss = nn.functional.mse_loss
                optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay = 1e-3)

                model.apply(init_weights)
                model.to(device)
                model = model.double()

                writer = SummaryWriter('runs/' + config + '_train')
                writer_valid = SummaryWriter('runs/' + config + '_valid')

                for epoch in tqdm(range(epochs)):
                    ### update theta  ###
                    for a, b in trainloader:
                        loss_pred, loss_fair = 0, 0
                        a = a.to(device)
                        b = b.to(device)

                        optimizer.zero_grad()
                        model.train()

                        a_00 = a[b==0][a[b == 0][:,sens_idx] == 1.]
                        a_01 = a[b==0][a[b == 0][:,sens_idx] == 2.]
                        a_10 = a[b==1][a[b == 1][:,sens_idx] == 1.]
                        a_11 = a[b==1][a[b == 1][:,sens_idx] == 2.]

                        b_00 = b[b==0][a[b == 0][:,sens_idx] == 1.]
                        b_01 = b[b==0][a[b == 0][:,sens_idx] == 2.]
                        b_10 = b[b==1][a[b == 1][:,sens_idx] == 1.]
                        b_11 = b[b==1][a[b == 1][:,sens_idx] == 2.]

                        if len(a_00) != 0:
                            #a_00_ref = a_00.clone()
                            #a_00_ref[:, sens_idx] = 1.5
                            #pred_00 = model(a_00_ref)
                            
                            pred_00 = model(a_00)
                            l_00 = celoss(pred_00, b_00, reduction='none')
                            l_00_sort, l_00_idx_train = l_00.detach().sort()

                        if len(a_01) != 0:
                            #a_01_ref = a_01.clone()
                            #a_01_ref[:, sens_idx] = 1.5
                            #pred_01 = model(a_01_ref)
                            
                            pred_01 = model(a_01)
                            l_01 = celoss(pred_01, b_01, reduction='none')
                            l_01_sort, l_01_idx_train = l_01.detach().sort()

                        if len(a_11) != 0:
                            #a_11_ref = a_11.clone()
                            #a_11_ref[:, sens_idx] = 1.5
                            #pred_11 = model(a_11_ref)
                            
                            pred_11 = model(a_11)
                            l_11 = celoss(pred_11, b_11, reduction='none')
                            l_11_sort, l_11_idx_train = l_11.detach().sort()

                        if len(a_10) != 0:
                            #a_10_ref = a_10.clone()
                            #a_10_ref[:, sens_idx] = 1.5
                            #pred_10 = model(a_10_ref)
                            
                            pred_10 = model(a_10)
                            l_10 = celoss(pred_10, b_10, reduction='none')
                            l_10_sort, l_10_idx_train = l_10.detach().sort()

                        with torch.no_grad():               

                            ## a = 1, y = 1 ##
                            if len(a_11) != 0:
                                checker = 0
                                c = int((len(a_11)+len(a_10))/2)
                                for i in range(1, len(l_11_sort)):
                                    if sum(l_11_sort[:i]) - i * l_11_sort[i] > 2 * alpha_p * c and \
                                    sum(l_11_sort[:i]) - i * l_11_sort[i-1] <= 2 * alpha_p * c:
                                        k = i
                                        checker = 1
                                        #print('epoch : {}, train l_11_idx : {} / {}'.format(epoch, k, len(l_11_sort)))
                                        break

                                if checker == 1:
                                    pass
                                else:
                                    #print('epoch : {}, len : {}, train l_11_idx : {:.4f} / {:.4f}'.format
                                    #      (epoch, len(l_11_sort), sum(l_11_sort[:i+1]) - (i+1) * l_11_sort[i], 
                                    #       2 * alpha_p * c))
                                    k = len(l_11_sort)

                                mu = (sum(l_11_sort[:k]) - 2 * alpha_p * c) / k

                                w_train_11 = (l_11_sort - mu)/(2*alpha_p)
                                w_train_11[k:] = 0
                            else:
                                pass

                            ## a = 0, y = 1 ##
                            if len(a_10) != 0:
                                checker = 0
                                c = int((len(a_11)+len(a_10))/2)
                                for i in range(1, len(l_10_sort)):
                                    if sum(l_10_sort[:i]) - i * l_10_sort[i] > 2 * alpha_p * c and \
                                    sum(l_10_sort[:i]) - i * l_10_sort[i-1] <= 2 * alpha_p * c:
                                        k = i
                                        checker = 1
                                        #print('epoch : {}, train l_10_idx : {} / {}'.format(epoch, k, len(l_10_sort)))
                                        break

                                if checker == 1:
                                    pass
                                else:
                                    #print('epoch : {}, len : {}, train l_10_idx : {:.4f} / {:.4f}'.format\
                                    #      (epoch, len(l_10_sort), sum(l_10_sort[:i+1]) - (i+1) * l_10_sort[i], 
                                    #       2 * alpha_p * c))
                                    k = len(l_10_sort)

                                mu = (sum(l_10_sort[:k]) - 2 * alpha_p * c) / k

                                w_train_10 = (l_10_sort - mu)/(2*alpha_p)
                                w_train_10[k:] = 0   
                            else:
                                pass


                            ## a = 1, y = 0 ##
                            if len(a_01) != 0:
                                checker = 0
                                c = int((len(a_01)+len(a_00))/2)
                                for i in range(1, len(l_01_sort)):
                                    if sum(l_01_sort[:i]) - i * l_01_sort[i] > 2 * alpha_n * c and \
                                    sum(l_01_sort[:i]) - i * l_01_sort[i-1] <= 2 * alpha_n * c:
                                        k = i
                                        checker = 1
                                        #print('epoch : {}, train l_01_idx : {} / {}'.format(epoch, k, len(l_01_sort)))
                                        break

                                if checker == 1:
                                    pass
                                else:
                                    #print('epoch : {}, len : {}, train l_01_idx : {:.4f} / {:.4f}'.format\
                                    #      (epoch, len(l_01_sort), sum(l_01_sort[:i+1]) - (i+1) * l_01_sort[i], 
                                    #       2 * alpha_n * c))
                                    k = len(l_01_sort)

                                mu = (sum(l_01_sort[:k]) - 2 * alpha_n * c) / k

                                w_train_01 = (l_01_sort - mu)/(2*alpha_n)
                                w_train_01[k:] = 0
                            else:
                                pass

                            ## a = 0, y = 0 ##
                            if len(a_00) != 0:
                                checker = 0
                                c = int((len(a_01)+len(a_00))/2)
                                for i in range(1, len(l_00_sort)):
                                    if sum(l_00_sort[:i]) - i * l_00_sort[i] > 2 * alpha_n * c and \
                                    sum(l_00_sort[:i]) - i * l_00_sort[i-1] <= 2 * alpha_n * c:
                                        k = i
                                        checker = 1
                                        #print('epoch : {}, train l_00_idx : {} / {}'.format(epoch, k, len(l_00_sort)))
                                        break

                                if checker == 1:
                                    pass
                                else:
                                    #print('epoch : {}, len : {}, train l_00_idx : {:.4f} / {:.4f}'.format\
                                    #      (epoch, len(l_00_sort), sum(l_00_sort[:i+1]) - (i+1) * l_00_sort[i], 
                                    #       2 * alpha_n * c))
                                    k = len(l_00_sort)

                                mu = (sum(l_00_sort[:k]) - 2 * alpha_n * c) / k

                                w_train_00 = (l_00_sort - mu)/(2*alpha_n)
                                w_train_00[k:] = 0
                            else:
                                pass

                        if len(a_10) != 0:
                            loss_pred_10 = (w_train_10 * l_10[l_10_idx_train]).mean()
                            loss_pred += loss_pred_10

                            f_10_priv = a_10.clone()
                            f_10_priv[:, sens_idx] = 2
                            p_10_priv = F.softmax(model(f_10_priv.double()))

                            f_10_unpriv = a_10.clone()
                            f_10_unpriv[:, sens_idx] = 1
                            p_10_unpriv = F.softmax(model(f_10_unpriv.double()))

                            fair_10 = mseloss(p_10_priv, p_10_unpriv, reduction = 'none').mean(-1)

                            loss_fair += (w_train_10 * fair_10[l_10_idx_train]).sum()

                        if len(a_11) != 0:
                            loss_pred_11 = (w_train_11 * l_11[l_11_idx_train]).mean()
                            loss_pred += loss_pred_11

                            f_11_priv = a_11.clone()
                            f_11_priv[:, sens_idx] = 2
                            p_11_priv = F.softmax(model(f_11_priv.double()))

                            f_11_unpriv = a_11.clone()
                            f_11_unpriv[:, sens_idx] = 1
                            p_11_unpriv = F.softmax(model(f_11_unpriv.double()))

                            fair_11 = mseloss(p_11_priv, p_11_unpriv, reduction = 'none').mean(-1)

                            loss_fair += (w_train_11 * fair_11[l_11_idx_train]).sum()

                        if len(a_00) != 0:
                            loss_pred_00 = (w_train_00 * l_00[l_00_idx_train]).mean()
                            loss_pred += loss_pred_00

                            f_00_priv = a_00.clone()
                            f_00_priv[:, sens_idx] = 2
                            p_00_priv = F.softmax(model(f_00_priv.double()))

                            f_00_unpriv = a_00.clone()
                            f_00_unpriv[:, sens_idx] = 1
                            p_00_unpriv = F.softmax(model(f_00_unpriv.double()))

                            fair_00 = mseloss(p_00_priv, p_00_unpriv, reduction = 'none').mean(-1)

                            loss_fair += (w_train_00 * fair_00[l_00_idx_train]).sum()


                        if len(a_01) != 0:
                            loss_pred_01 = (w_train_01 * l_01[l_01_idx_train]).mean()
                            loss_pred += loss_pred_01

                            f_01_priv = a_01.clone()
                            f_01_priv[:, sens_idx] = 2
                            p_01_priv = F.softmax(model(f_01_priv.double()))

                            f_01_unpriv = a_01.clone()
                            f_01_unpriv[:, sens_idx] = 1
                            p_01_unpriv = F.softmax(model(f_01_unpriv.double()))

                            fair_01 = mseloss(p_01_priv, p_01_unpriv, reduction = 'none').mean(-1)

                            loss_fair += (w_train_01 * fair_01[l_01_idx_train]).sum()

                        loss = (loss_pred) + (beta * loss_fair)
                        loss.backward()
                        optimizer.step()

                        pred = model(a)

                        acc = sum(pred.argmax(-1) == b)/float(len(b))

                        #print("epoch : {0:d} loss : {1:4f}, acc : {2:4f}".format(epoch, loss, acc))
                    
                    hist['loss'].append(loss)
                    hist['acc'].append(acc)

                    writer.add_scalar('weighted_loss', loss_pred.item(), epoch)
                    writer.add_scalar('loss_fair', loss_fair.item(), epoch)
                    writer.add_scalar('loss', loss.item(), epoch)
                    writer.add_scalar('acc', acc, epoch)

                    #writer.add_histogram('w_train_1', w_train_1, epoch)
                    #writer.add_histogram('w_train_0', w_train_0, epoch)

                    ### eval network ###

                    model = model.eval()

                    with torch.no_grad():
                        cnt, loss_pred, loss_fair, acc = 0, 0, 0, 0
                        for a,b in validloader:
                            cnt += 1                             

                            loss_pred, loss_fair = 0, 0
                            a = a.to(device)
                            b = b.to(device)

                            a_00 = a[b==0][a[b == 0][:,sens_idx] == 1.]
                            a_01 = a[b==0][a[b == 0][:,sens_idx] == 2.]
                            a_10 = a[b==1][a[b == 1][:,sens_idx] == 1.]
                            a_11 = a[b==1][a[b == 1][:,sens_idx] == 2.]

                            b_00 = b[b==0][a[b == 0][:,sens_idx] == 1.]
                            b_01 = b[b==0][a[b == 0][:,sens_idx] == 2.]
                            b_10 = b[b==1][a[b == 1][:,sens_idx] == 1.]
                            b_11 = b[b==1][a[b == 1][:,sens_idx] == 2.]

                            if len(a_00) != 0:
                                #a_00_ref = a_00.clone()
                                #a_00_ref[:, sens_idx] = 1.5
                                #pred_00 = model(a_00_ref)
                                
                                pred_00 = model(a_00)
                                l_00 = celoss(pred_00, b_00, reduction='none')
                                l_00_sort, l_00_idx_valid = l_00.detach().sort()

                            if len(a_01) != 0:
                                #a_01_ref = a_01.clone()
                                #a_01_ref[:, sens_idx] = 1.5
                                #pred_01 = model(a_01_ref)
                                
                                pred_01 = model(a_01)
                                l_01 = celoss(pred_01, b_01, reduction='none')
                                l_01_sort, l_01_idx_valid = l_01.detach().sort()

                            if len(a_11) != 0:
                                #a_11_ref = a_11.clone()
                                #a_11_ref[:, sens_idx] = 1.5
                                #pred_11 = model(a_11_ref)
                                
                                pred_11 = model(a_11)
                                l_11 = celoss(pred_11, b_11, reduction='none')
                                l_11_sort, l_11_idx_valid = l_11.detach().sort()

                            if len(a_10) != 0:
                                #a_10_ref = a_10.clone()
                                #a_10_ref[:, sens_idx] = 1.5
                                #pred_10 = model(a_10_ref)
                                
                                pred_10 = model(a_10)
                                l_10 = celoss(pred_10, b_10, reduction='none')
                                l_10_sort, l_10_idx_valid = l_10.detach().sort()

                            ## a = 1, y = 1 ##
                            if len(a_11) != 0:
                                checker = 0
                                c = int((len(a_11)+len(a_10))/2)
                                for i in range(1, len(l_11_sort)):
                                    if sum(l_11_sort[:i]) - i * l_11_sort[i] > 2 * alpha_p * c and\
                                    sum(l_11_sort[:i]) - i * l_11_sort[i-1] <= 2 * alpha_p * c:
                                        k = i
                                        checker = 1
                                        #print('epoch : {}, valid l_11_idx : {} / {}'.format(epoch, k, len(l_11_sort)))
                                        break

                                if checker == 1:
                                    pass
                                else:
                                    #print('epoch : {}, len : {}, valid l_11_idx : {:.4f} / {:.4f}'.format\
                                   #       (epoch, len(l_11_sort), sum(l_11_sort[:i+1]) - (i+1) * l_11_sort[i], 
                                    #       2 * alpha_p * c))
                                    k = len(l_11_sort)

                                mu = (sum(l_11_sort[:k]) - 2 * alpha_p * c) / k

                                w_valid_11 = (l_11_sort - mu)/(2*alpha_p)
                                w_valid_11[k:] = 0

                                loss_pred_11 = (w_valid_11 * l_11_sort).mean()
                                loss_pred += loss_pred_11

                                f_11_priv = a_11.clone()
                                f_11_priv[:, sens_idx] = 2
                                p_11_priv = F.softmax(model(f_11_priv.double()))

                                f_11_unpriv = a_11.clone()
                                f_11_unpriv[:, sens_idx] = 1
                                p_11_unpriv = F.softmax(model(f_11_unpriv.double()))

                                fair_11 = mseloss(p_11_priv, p_11_unpriv, reduction = 'none').mean(-1)

                                loss_fair += (w_valid_11 * fair_11[l_11_idx_valid]).sum()


                            ## a = 0, y = 1 ##
                            if len(a_10)!= 0:
                                checker = 0
                                c = int((len(a_11)+len(a_10))/2)
                                for i in range(1, len(l_10_sort)):
                                    if sum(l_10_sort[:i]) - i * l_10_sort[i] > 2 * alpha_p * c and\
                                    sum(l_10_sort[:i]) - i * l_10_sort[i-1] <= 2 * alpha_p * c:
                                        k = i
                                        checker = 1
                                        #print('epoch : {}, valid l_10_idx : {} / {}'.format(epoch, k, len(l_10_sort)))
                                        break

                                if checker == 1:
                                    pass
                                else:
                                    #print('epoch : {}, len : {}, valid l_10_idx : {:.4f} / {:.4f}'.format\
                                    #      (epoch, len(l_10_sort), sum(l_10_sort[:i+1]) - (i+1) * l_10_sort[i], 
                                    #       2 * alpha_p * c))
                                    k = len(l_10_sort)

                                mu = (sum(l_10_sort[:k]) - 2 * alpha_p * c) / k

                                w_valid_10 = (l_10_sort - mu)/(2*alpha_p)
                                w_valid_10[k:] = 0   

                                loss_pred_10 = (w_valid_10 * l_10_sort).mean()
                                loss_pred += loss_pred_10

                                f_10_priv = a_10.clone()
                                f_10_priv[:, sens_idx] = 2
                                p_10_priv = F.softmax(model(f_10_priv.double()))

                                f_10_unpriv = a_10.clone()
                                f_10_unpriv[:, sens_idx] = 1
                                p_10_unpriv = F.softmax(model(f_10_unpriv.double()))

                                fair_10 = mseloss(p_10_priv, p_10_unpriv, reduction = 'none').mean(-1)

                                loss_fair += (w_valid_10 * fair_10[l_10_idx_valid]).sum()


                            ## a = 1, y = 0 ##
                            if len(a_01) != 0:
                                checker = 0
                                c = int((len(a_00)+len(a_01))/2)
                                for i in range(1, len(l_01_sort)):
                                    if sum(l_01_sort[:i]) - i * l_01_sort[i] > 2 * alpha_n * c and\
                                    sum(l_01_sort[:i]) - i * l_01_sort[i-1] <= 2 * alpha_n * c:
                                        k = i
                                        checker = 1
                                        #print('epoch : {}, valid l_01_idx : {} / {}'.format(epoch, k, len(l_01_sort)))
                                        break

                                if checker == 1:
                                    pass
                                else:
                                    #print('epoch : {}, len : {}, valid l_01_idx : {:.4f} / {:.4f}'.format\
                                    #      (epoch, len(l_01_sort), sum(l_01_sort[:i+1]) - (i+1) * l_01_sort[i], 
                                    #       2 * alpha_n * c))
                                    k = len(l_01_sort)

                                mu = (sum(l_01_sort[:k]) - 2 * alpha_n * c) / k

                                w_valid_01 = (l_01_sort - mu)/(2*alpha_n)
                                w_valid_01[k:] = 0

                                loss_pred_01 = (w_valid_01 * l_01_sort).mean()
                                loss_pred += loss_pred_01

                                f_01_priv = a_01.clone()
                                f_01_priv[:, sens_idx] = 2
                                p_01_priv = F.softmax(model(f_01_priv.double()))

                                f_01_unpriv = a_01.clone()
                                f_01_unpriv[:, sens_idx] = 1
                                p_01_unpriv = F.softmax(model(f_01_unpriv.double()))

                                fair_01 = mseloss(p_01_priv, p_01_unpriv, reduction = 'none').mean(-1)

                                loss_fair += (w_valid_01 * fair_01[l_01_idx_valid]).sum()

                            ## a = 0, y = 0 ##
                            if len(a_00) != 0:
                                checker = 0
                                c = int((len(a_00)+len(a_01))/2)
                                for i in range(1, len(l_00_sort)):
                                    if sum(l_00_sort[:i]) - i * l_00_sort[i] > 2 * alpha_n * c and\
                                    sum(l_00_sort[:i]) - i * l_00_sort[i-1] <= 2 * alpha_n * c:
                                        k = i
                                        checker = 1
                                        #print('epoch : {}, valid l_00_idx : {} / {}'.format(epoch, k, len(l_00_sort)))
                                        break

                                if checker == 1:
                                    pass
                                else:
                                    #print('epoch : {}, len : {}, valid l_00_idx : {:.4f} / {:.4f}'.format\
                                    #      (epoch, len(l_00_sort), sum(l_00_sort[:i+1]) - (i+1) * l_00_sort[i], 
                                    #       2 * alpha_n * c))
                                    k = len(l_00_sort)

                                mu = (sum(l_00_sort[:k]) - 2 * alpha_n * c) / k

                                w_valid_00 = (l_00_sort - mu)/(2*alpha_n)
                                w_valid_00[k:] = 0

                                loss_pred_00 = (w_valid_00 * l_00_sort).mean()
                                loss_pred += loss_pred_00

                                f_00_priv = a_00.clone()
                                f_00_priv[:, sens_idx] = 2
                                p_00_priv = F.softmax(model(f_00_priv.double()))

                                f_00_unpriv = a_00.clone()
                                f_00_unpriv[:, sens_idx] = 1
                                p_00_unpriv = F.softmax(model(f_00_unpriv.double()))

                                fair_00 = mseloss(p_00_priv, p_00_unpriv, reduction = 'none').mean(-1)

                                loss_fair += (w_valid_00 * fair_00[l_00_idx_valid]).sum()

                            pred = model(a)
                            acc += (sum(pred.argmax(-1) == b)/float(len(b))).item()

                        loss = (loss_pred + beta * loss_fair)/cnt
                        acc /= cnt
                        loss_fair /= cnt

                        writer_valid.add_scalar('weighted_loss', loss_pred)
                        writer_valid.add_scalar('loss_fair', loss_fair)
                        writer_valid.add_scalar('loss', loss)
                        writer_valid.add_scalar('acc', acc)

                        hist['val_loss'].append(loss)
                        hist['val_acc'].append(acc)



                    if acc < 0.43 and epoch > 50:
                        best_epoch = epoch
                        break

                    #if acc > best_acc and epoch > 5:
                    if best_loss > loss.item() and epoch > 5:
                        print(epoch)
                        best_loss = loss.item()
                        best_acc = acc
                        model_best = copy.deepcopy(model)
                        best_epoch = epoch
                        cnt = 0

                        tpr_overall, tpr_priv, tpr_unpriv,\
                        fpr_overall, fpr_unpriv, fpr_priv,\
                        acc_overall, acc_priv, acc_unpriv, eq_overall = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

                        with torch.no_grad():
                            for a, b in validloader:
                                cnt += 1
                                a = a.to(device)

                                test_priv = a[a[:,sens_idx] == 2.]
                                test_unpriv = a[a[:,sens_idx] == 1.]

                                test_lb_priv = b[a[:,sens_idx] == 2.]
                                test_lb_unpriv = b[a[:,sens_idx] == 1.]

                                pred_test = model(a)
                                pred_priv = model(test_priv)
                                pred_unpriv = model(test_unpriv)

                                b = b.cpu().detach().numpy()
                                test_lb_priv = test_lb_priv.cpu().detach().numpy()
                                test_lb_unpriv = test_lb_unpriv.cpu().detach().numpy()

                                pred_test = pred_test.cpu().detach().numpy().argmax(1)
                                pred_priv = pred_priv.cpu().detach().numpy().argmax(1)
                                pred_unpriv = pred_unpriv.cpu().detach().numpy().argmax(1)

                                tpr_overall = sklm.recall_score(b, pred_test) #act recidivism, detection rate.
                                tpr_priv = sklm.recall_score(test_lb_priv, pred_priv) #act recidivism, detection rate.
                                tpr_unpriv = sklm.recall_score(test_lb_unpriv, pred_unpriv) #act recidivism, detection rate.

                                fpr_overall += fpr_calc(b, pred_test) #act recidivism, detection rate.
                                fpr_priv += fpr_calc(test_lb_priv, pred_priv) #act recidivism, detection rate.
                                fpr_unpriv += fpr_calc(test_lb_unpriv, pred_unpriv) #act recidivism, detection rate.

                                acc_overall += sklm.accuracy_score(b, pred_test)
                                acc_priv += sklm.accuracy_score(test_lb_priv, pred_priv)
                                acc_unpriv += sklm.accuracy_score(test_lb_unpriv, pred_unpriv)

                                eq_overall += abs(tpr_priv - tpr_unpriv)

                            tpr_overall /= cnt
                            tpr_priv /= cnt
                            tpr_unpriv /= cnt
                            fpr_overall /= cnt
                            fpr_unpriv /= cnt
                            fpr_priv /= cnt
                            acc_overall /= cnt
                            acc_priv /= cnt
                            acc_unpriv /= cnt
                            eq_overall /= cnt

                            print('\n overall : ', data_train.labels.shape[0])
                            print()
                            print('overall TPR : {0:.3f}'.format( tpr_overall))
                            print('priv TPR : {0:.3f}'.format( tpr_priv))
                            print('unpriv TPR : {0:.3f}'.format( tpr_unpriv))
                            print('Eq. Opp : {0:.3f}'.format( eq_overall))
                            print()
                            print('overall FPR : {0:.3f}'.format( fpr_overall))
                            print('priv FPR : {0:.3f}'.format( fpr_priv))
                            print('unpriv FPR : {0:.3f}'.format( fpr_unpriv))
                            print('diff FPR : {0:.3f}'.format( abs(fpr_unpriv-fpr_priv)))
                            print()
                            print('overall ACC : {0:.3f}'.format( acc_overall))
                            print('priv ACC : {0:.3f}'.format( acc_priv))
                            print('unpriv ACC : {0:.3f}'.format( acc_unpriv)) 
                            print('diff ACC : {0:.3f}'.format( abs(acc_unpriv-acc_priv)))

                    elif (epoch+1)%50 == 0:
                        print(epoch+1)
                        cnt = 0

                        tpr_overall, tpr_priv, tpr_unpriv,\
                        fpr_overall, fpr_unpriv, fpr_priv,\
                        acc_overall, acc_priv, acc_unpriv, eq_overall = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

                        with torch.no_grad():
                            for a, b in validloader:
                                cnt += 1
                                a = a.to(device)

                                test_priv = a[a[:,sens_idx] == 2.]
                                test_unpriv = a[a[:,sens_idx] == 1.]

                                test_lb_priv = b[a[:,sens_idx] == 2.]
                                test_lb_unpriv = b[a[:,sens_idx] == 1.]

                                pred_test = model(a)
                                pred_priv = model(test_priv)
                                pred_unpriv = model(test_unpriv)

                                b = b.cpu().detach().numpy()
                                test_lb_priv = test_lb_priv.cpu().detach().numpy()
                                test_lb_unpriv = test_lb_unpriv.cpu().detach().numpy()

                                pred_test = pred_test.cpu().detach().numpy().argmax(1)
                                pred_priv = pred_priv.cpu().detach().numpy().argmax(1)
                                pred_unpriv = pred_unpriv.cpu().detach().numpy().argmax(1)

                                tpr_overall += sklm.recall_score(b, pred_test) #act recidivism, detection rate.
                                tpr_priv += sklm.recall_score(test_lb_priv, pred_priv) #act recidivism, detection rate.
                                tpr_unpriv += sklm.recall_score(test_lb_unpriv, pred_unpriv) #act recidivism, detection rate.

                                fpr_overall += fpr_calc(b, pred_test) #act recidivism, detection rate.
                                fpr_priv += fpr_calc(test_lb_priv, pred_priv) #act recidivism, detection rate.
                                fpr_unpriv += fpr_calc(test_lb_unpriv, pred_unpriv) #act recidivism, detection rate.

                                acc_overall += sklm.accuracy_score(b, pred_test)
                                acc_priv += sklm.accuracy_score(test_lb_priv, pred_priv)
                                acc_unpriv += sklm.accuracy_score(test_lb_unpriv, pred_unpriv)

                                eq_overall += abs(tpr_priv - tpr_unpriv)

                            tpr_overall /= cnt
                            tpr_priv /= cnt
                            tpr_unpriv /= cnt
                            fpr_overall /= cnt
                            fpr_unpriv /= cnt
                            fpr_priv /= cnt
                            acc_overall /= cnt
                            acc_priv /= cnt
                            acc_unpriv /= cnt
                            eq_overall /= cnt                          

                            print('\n overall : ', data_train.labels.shape[0])
                            print()
                            print('overall TPR : {0:.3f}'.format( tpr_overall))
                            print('priv TPR : {0:.3f}'.format( tpr_priv))
                            print('unpriv TPR : {0:.3f}'.format( tpr_unpriv))
                            print('Eq. Opp : {0:.3f}'.format( eq_overall))
                            print()
                            print('overall FPR : {0:.3f}'.format( fpr_overall))
                            print('priv FPR : {0:.3f}'.format( fpr_priv))
                            print('unpriv FPR : {0:.3f}'.format( fpr_unpriv))
                            print('diff FPR : {0:.3f}'.format( abs(fpr_unpriv-fpr_priv)))
                            print()
                            print('overall ACC : {0:.3f}'.format( acc_overall))
                            print('priv ACC : {0:.3f}'.format( acc_priv))
                            print('unpriv ACC : {0:.3f}'.format( acc_unpriv)) 
                            print('diff ACC : {0:.3f}'.format( abs(acc_unpriv-acc_priv)))

                if epoch < epochs-1:
                    continue
                else:
                    torch.save(model_best.state_dict(), 'save/'+ config + '.pth')

                    plt.plot(hist['val_loss'])
                    plt.plot(hist['loss'], '--')
                    plt.ylabel( 'loss' ) 
                    plt.xlabel( 'Epoch') 
                    plt.legend([ 'overall_val','overall_train'] , loc='upper right' , bbox_to_anchor=(1.5, 0.5))
                    plt.savefig('plt_loss_{:5f}_{}_{}.jpg'.format(lr, beta, alpha))

                    plt.plot(hist['val_acc'])
                    plt.plot(hist['acc'], '--')
                    plt.ylabel( 'ACC' ) 
                    plt.xlabel( 'Epoch') 
                    plt.legend([ 'overall_val','overall_train'] , loc='upper right' , bbox_to_anchor=(1.5, 0.5))
                    plt.savefig('plt_acc_{:5f}_{}_{}.jpg'.format(lr, beta, alpha))








