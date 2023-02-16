import torch
import numpy as np
import torch.utils.data as data

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

def max_margin_loss(y_onehot, y_pred, alpha, mu, lamda, device):
    
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

def data_copy(dataset, priv_idx, label_idx, rand_idx):
    data_new = dataset.copy(deepcopy=True)

    data_new.instance_weights  = dataset.instance_weights[priv_idx][label_idx][rand_idx]
    data_new.protected_attributes = dataset.protected_attributes[priv_idx][label_idx][rand_idx]
    data_new.features = dataset.features[priv_idx][label_idx][rand_idx]
    data_new.labels = dataset.labels[priv_idx][label_idx][rand_idx]
    data_new.scores = dataset.scores[priv_idx][label_idx][rand_idx]
    data_new.instance_names = list(np.array(dataset.instance_names)[priv_idx][label_idx][rand_idx])

    return data_new

def upsample_all_equal(dataset, sens_idx):     
    data_train, data_vt = dataset.split([0.7], shuffle=True)
    data_valid, data_test = data_vt.split([0.5], shuffle=True)
    
    num_samples = {}

    num_sens    = len(sens_idx)

    combination = [list(i) for i in itertools.product([0, 1], repeat=num_sens)]

    for lb in [0,1]:
        lb_idx = (data_train.labels == lb).reshape(-1)
        for comb in combination:
            comb_str = str(lb)
            num = 1
            for i in range(num_sens):
                num *= data_train.features[lb_idx][:, sens_idx[i]] == comb[i]
                comb_str += str(comb[i])
            num_samples['{}'.format(comb_str)] = sum(num), num.astype(bool)
            
    all_values = num_samples.values()
    max_value = max(all_values)[0]
    
    cnt = 1
    for key, value in num_samples.items():
        if value[0] != max_value:
            lb = int(key[0])
            lb_idx = (data_train.labels == lb).reshape(-1)
            rand_idx = np.random.randint(1,value[0], max_value)
            
            if cnt == 1:
                data_new = data_copy(data_train, lb_idx, value[1], rand_idx)
            else:
                data_tmp = data_copy(data_train, lb_idx, value[1], rand_idx)
                data_new.features = np.concatenate((data_new.features, data_tmp.features), 0)
                data_new.labels = np.concatenate((data_new.labels, data_tmp.labels), 0)
                data_new.instance_weights = np.concatenate((data_new.instance_weights, data_tmp.instance_weights), 0)
                data_new.protected_attributes = np.concatenate((data_new.protected_attributes, data_tmp.protected_attributes), 0)
                data_new.scores = np.concatenate((data_new.scores, data_tmp.scores), 0)

                data_new.instance_names = list(np.concatenate((data_new.instance_names, data_tmp.instance_names), 0))
                
        else:
            lb = int(key[0])
            lb_idx = (data_train.labels == lb).reshape(-1)
            
            if cnt == 1:
                data_new = data_copy(data_train,lb_idx, value[1], range(max_value))
            else:
                data_tmp = data_copy(data_train,lb_idx, value[1], range(max_value))
                data_new.features = np.concatenate((data_new.features, data_tmp.features), 0)
                data_new.labels = np.concatenate((data_new.labels, data_tmp.labels), 0)
                data_new.instance_weights = np.concatenate((data_new.instance_weights, data_tmp.instance_weights), 0)
                data_new.protected_attributes = np.concatenate((data_new.protected_attributes, data_tmp.protected_attributes), 0)
                data_new.scores = np.concatenate((data_new.scores, data_tmp.scores), 0)

                data_new.instance_names = list(np.concatenate((data_new.instance_names, data_tmp.instance_names), 0))

        cnt += 1    
        
    print(num_samples)
         
    return data_new, data_valid, data_test


def downsample_all_equal(dataset, sens_idx):     
    data_train, data_vt = dataset.split([0.7], shuffle=True)
    data_valid, data_test = data_vt.split([0.5], shuffle=True)
    
    num_samples = {}

    num_sens    = len(sens_idx)

    combination = [list(i) for i in itertools.product([0, 1], repeat=num_sens)]

    for lb in [0,1]:
        lb_idx = (data_train.labels == lb).reshape(-1)
        for comb in combination:
            comb_str = str(lb)
            num = 1
            for i in range(num_sens):
                num *= data_train.features[lb_idx][:, sens_idx[i]] == comb[i]
                comb_str += str(comb[i])
            num_samples['{}'.format(comb_str)] = sum(num), num.astype(bool)
            
    all_values = num_samples.values()
    min_value = min(all_values)[0]
    
    cnt = 1
    for key, value in num_samples.items():
        if value[0] != min_value:
            lb = int(key[0])
            lb_idx = (data_train.labels == lb).reshape(-1)
            rand_idx = np.random.randint(1,value[0], min_value)
            
            if cnt == 1:
                data_new = data_copy(data_train, lb_idx, value[1], rand_idx)
            else:
                data_tmp = data_copy(data_train, lb_idx, value[1], rand_idx)
                data_new.features = np.concatenate((data_new.features, data_tmp.features), 0)
                data_new.labels = np.concatenate((data_new.labels, data_tmp.labels), 0)
                data_new.instance_weights = np.concatenate((data_new.instance_weights, data_tmp.instance_weights), 0)
                data_new.protected_attributes = np.concatenate((data_new.protected_attributes, data_tmp.protected_attributes), 0)
                data_new.scores = np.concatenate((data_new.scores, data_tmp.scores), 0)

                data_new.instance_names = list(np.concatenate((data_new.instance_names, data_tmp.instance_names), 0))
                
        else:
            lb = int(key[0])
            lb_idx = (data_train.labels == lb).reshape(-1)
            
            if cnt == 1:
                data_new = data_copy(data_train,lb_idx, value[1], range(min_value))
            else:
                data_tmp = data_copy(data_train,lb_idx, value[1], range(min_value))
                data_new.features = np.concatenate((data_new.features, data_tmp.features), 0)
                data_new.labels = np.concatenate((data_new.labels, data_tmp.labels), 0)
                data_new.instance_weights = np.concatenate((data_new.instance_weights, data_tmp.instance_weights), 0)
                data_new.protected_attributes = np.concatenate((data_new.protected_attributes, data_tmp.protected_attributes), 0)
                data_new.scores = np.concatenate((data_new.scores, data_tmp.scores), 0)

                data_new.instance_names = list(np.concatenate((data_new.instance_names, data_tmp.instance_names), 0))

        cnt += 1    
        
    print(num_samples)
         
    return data_new, data_valid, data_test


