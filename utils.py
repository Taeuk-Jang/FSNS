import sys
import os
import numpy as np
import torch
import torch.utils.data as data
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Dataset(data.Dataset):
    def __init__(self, dataset, sens_idx):
        self.label = dataset.labels.squeeze(-1).astype(int)
        
        self.feature_size = dataset.features.shape[1]
        sens_loc = np.zeros(self.feature_size).astype(bool)
        sens_loc[sens_idx] = 1

        self.feature = dataset.features[:,~sens_loc] #data without sensitive
#         self.feature = min_max_scaler.fit_transform(self.feature)
        
        self.sensitive = dataset.features[:,sens_loc].reshape(-1).astype(int)
        #n_values = int(np.max(self.label) + 1)
        #self.label = np.eye(n_values)[self.label.astype(int)].squeeze(1)
        
    def __getitem__(self, idx):
    
        y = self.label[idx]
        x = self.feature[idx] + np.random.normal(0, 1e-6, self.feature_size - 1)
        a = self.sensitive[idx]
        
        return x, a, y
    
    
    def __len__(self):
        return len(self.label)
    
def fpr_calc(label, pred):
    FP = (sum(pred[(label == 0)])) #False Positive
    TN = (len(pred[(label == 0)]) - sum(pred[(label == 0)]))

    if FP+TN == 0:
        fpr = 0
    else:
        fpr = FP / (FP+TN)
    
    return fpr

def tpr_calc(label, pred):
    TP = (sum(pred[(label == 1)])) #False Positive
    FN = (len(pred[(label == 1)]) - sum(pred[(label == 1)]))

    if TP+FN == 0:
        tpr = 0
    else:
        tpr = TP / (TP+FN)
    
    return tpr


class Feature_extractor(nn.Module):
    def __init__(self, input_size, hidden_size = 256, latent_size = 50):
        super(Feature_extractor, self).__init__()

        self.dense1 = nn.Linear(input_size, hidden_size)
        #self.bn1 = nn.BatchNorm1d(hidden_size)

        self.dense2 = nn.Linear(hidden_size, hidden_size)
        #self.bn2 = nn.BatchNorm1d(hidden_size)

        self.dense3 = nn.Linear(hidden_size, hidden_size)       
        #self.bn3 = nn.BatchNorm1d(hidden_size)

        self.dense4 = nn.Linear(hidden_size, latent_size)
        #self.bn4 = nn.BatchNorm1d(hidden_size)

        self.dropout = nn.Dropout(0.3)

    def forward(self, x):        
        x = F.leaky_relu((self.dense1(x)), 0.1)

        x = (F.leaky_relu((self.dense2(x)), 0.1))

        x = (F.leaky_relu((self.dense3(x)), 0.1))

        x = F.leaky_relu(self.dense4(x),0.1)

        return x 
    
class Feature_predictor(nn.Module):
    def __init__(self, input_size, hidden_size = 256, latent_size = 50):
        super(Feature_predictor, self).__init__()
        
        self.dense1 = nn.Linear(input_size, hidden_size)
        
        self.dense2 = nn.Linear(hidden_size, hidden_size)
        
#         self.dense3 = nn.Linear(hidden_size, hidden_size)       
        
#         self.dense4 = nn.Linear(hidden_size, hidden_size)
        
        self.dense5 = nn.Linear(hidden_size, hidden_size)
        
        self.dense6 = nn.Linear(hidden_size, latent_size)
        
        
        
    def forward(self, x):        
        x = F.leaky_relu(self.dense1(x), 0.1)

        x = F.leaky_relu(self.dense2(x), 0.1)

#         x = F.leaky_relu(self.dense3(x), 0.1)
        
#         x = F.leaky_relu(self.dense4(x), 0.1)
        
        x = F.leaky_relu(self.dense5(x), 0.1)
        
#         x = F.softmax(self.dense6(x), dim= -1)
        x = self.dense6(x)
        
        return x 
    
class Cls_layer(nn.Module):
    def __init__(self, input_size, latent_size = 50):
        super(Cls_layer, self).__init__()
        
        self.dense = nn.Linear(input_size, latent_size)       
        
        
    def forward(self, x):        

        x = F.softmax(self.dense(x))
        
        return x 
    
    
def one_hot_embedding(labels, num_classes):
    y = torch.eye(num_classes) 
    return y[labels] 

def gamma(y_onehot, y_pred):
    idx = y_onehot == 1   
    
    if y_pred.shape[-1] > 2:
        margin = y_pred[idx] - torch.max(y_pred[~idx], dim = -1)
    else:
        margin = y_pred[idx] - y_pred[~idx]
    return margin
    
def svm_loss(target_onehot, output): 
    return torch.mean(torch.clamp((1 - (target_onehot * output)[target_onehot==1]), min = 0))

def max_out(input):
    return torch.mean(torch.clamp(input, min = 0))

def max_margin_loss(y_onehot, y_pred, alpha, mu):
    loss = (max_out(1 - alpha - gamma(y_onehot, y_pred))**2 + mu * max_out(gamma(y_onehot, y_pred) - 1 - alpha)**2)/(1 - alpha)**2
    return loss

def max_margin_loss_v2(y_onehot, y_pred, alpha, mu, lamda = 1):
    
    margin = y_pred[y_onehot==1] - y_pred[y_onehot==0]
    margin_mean = torch.mean(margin)
    
    
    xi =  torch.zeros(y_pred.shape[0]).to(device)
    epsilon =  torch.zeros(y_pred.shape[0]).to(device)
    
    xi = margin_mean - alpha - margin
    epsilon = margin - margin_mean - alpha

    margin_var = torch.mean(lamda/y_pred.shape[0] * (xi**2 + mu * epsilon ** 2)/((1-alpha)**2))
    
    return -margin_mean + margin_var


def cross_entropy(target_onehot, pred):
    return -torch.mean((target_onehot * torch.log(pred + 1e-5)))



def data_gen(dataset, sens_attr, rep_l=0, rep_s=0, label=0, sens=0, bs = 256):
    sens_idx = dataset.feature_names.index(sens_attr)
        
    data_train, data_vt = dataset.split([0.7], shuffle=True)
    data_valid, data_test = data_vt.split([0.5], shuffle=True)
    print(data_train.features.shape)
    orig_size = data_train.labels.shape[0]
    
    if rep_l > 0:
        data_new = data_train.copy(deepcopy=True)
        copy_idx = (data_train.labels == label).squeeze(-1)

        instance_weights  = data_train.instance_weights[copy_idx]
        protected_attributes = data_train.protected_attributes[copy_idx]
        sens_data = data_train.features[copy_idx]
        sens_label = data_train.labels[copy_idx]
        sens_scores = data_train.scores[copy_idx]
        instance_names = np.array(data_train.instance_names)[copy_idx]

        data_new.features = np.concatenate((data_new.features, np.repeat(sens_data, rep_l, axis = 0)), 0)
        data_new.labels = np.concatenate((data_new.labels, np.repeat(sens_label, rep_l, axis = 0)), 0)
        data_new.instance_weights = np.concatenate((data_new.instance_weights, np.repeat(instance_weights, rep_l, axis = 0)), 0)
        data_new.protected_attributes = np.concatenate((data_new.protected_attributes, np.repeat(protected_attributes, rep_l, axis = 0)), 0)
        data_new.scores = np.concatenate((data_new.scores, np.repeat(sens_scores, rep_l, axis = 0)), 0)

        data_new.instance_names = list(np.concatenate((np.array(data_new.instance_names), np.repeat(instance_names, rep_l, axis = 0)), 0))

        data_train = data_new.copy(deepcopy=True)
    
    if rep_s > 0:
        data_new = data_train.copy(deepcopy=True)
        copy_idx = (data_train.features[:, sens_idx] == sens)

        instance_weights  = data_train.instance_weights[copy_idx]
        protected_attributes = data_train.protected_attributes[copy_idx]
        sens_data = data_train.features[copy_idx]
        sens_label = data_train.labels[copy_idx]
        sens_scores = data_train.scores[copy_idx]
        instance_names = np.array(data_train.instance_names)[copy_idx]

        data_new.features = np.concatenate((data_new.features, np.repeat(sens_data, rep_s, axis = 0)), 0)
        data_new.labels = np.concatenate((data_new.labels, np.repeat(sens_label, rep_s, axis = 0)), 0)
        data_new.instance_weights = np.concatenate((data_new.instance_weights, np.repeat(instance_weights, rep_s, axis = 0)), 0)
        data_new.protected_attributes = np.concatenate((data_new.protected_attributes, np.repeat(protected_attributes, rep_s, axis = 0)), 0)
        data_new.scores = np.concatenate((data_new.scores, np.repeat(sens_scores, rep_s, axis = 0)), 0)

        data_new.instance_names = list(np.concatenate((np.array(data_new.instance_names), np.repeat(instance_names, rep_s, axis = 0)), 0))

        data_train = data_new.copy(deepcopy=True)   
     
    #data_train, _ = data_train.split([orig_size/data_train.labels.shape[0]], shuffle = True)
    print(data_train.features.shape)
    d_train = Dataset(data_train, sens_idx)
    d_valid = Dataset(data_valid, sens_idx)
    d_test = Dataset(data_test, sens_idx)


    #bs_train, bs_valid, bs_test = len(d_train), len(d_valid), len(d_test)
    bs_train, bs_valid, bs_test = 256, 256, 256
    # bs_train, bs_valid, bs_test = 64, 64, 64

    trainloader = torch.utils.data.DataLoader(
        d_train,
        batch_size=bs_train,
        shuffle=True,
        num_workers=16)

    validloader = torch.utils.data.DataLoader(
        d_valid,
        batch_size=bs_valid,
        shuffle=True,
        num_workers=16)
    testloader = torch.utils.data.DataLoader(
        d_test,
        batch_size=bs_test,
        shuffle=True,
        num_workers=16)
    
    return data_train, data_valid, data_test, trainloader, validloader, testloader
    
    
def max_margin_loss_v3(y_onehot, y_pred, alpha, mu, lamda = 1, weight = 1):
    device = y_onehot.device
    
    margin = y_pred[y_onehot==1] - y_pred[y_onehot==0]
    margin_mean = torch.mean(margin)
    
    xi =  torch.zeros(y_pred.shape[0]).to(device)
    epsilon =  torch.zeros(y_pred.shape[0]).to(device)
    
#     xi = margin_mean - alpha - margin
#     epsilon = margin - margin_mean - alpha
    xi = 1 - alpha - margin
    epsilon = margin - 1 - alpha

    margin_var = lamda/y_pred.shape[0] * (xi**2 + mu * epsilon ** 2)/((1-alpha)**2)
    
    return (weight *(-margin + margin_var)).mean()

def get_weights(a, y):
    weight = torch.zeros_like(y).double()
    weight[(a==0)*(y==0)] += sum((a==1)*(y==1))
    weight[(a==0)*(y==1)] += sum((a==1)*(y==0))
    weight[(a==1)*(y==0)] += sum((a==0)*(y==1))
    weight[(a==1)*(y==1)] += sum((a==0)*(y==0))
    weight /= y.shape[0]
    return weight