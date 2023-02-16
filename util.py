import torch
import torch.utils.data as data

class Dataset(data.Dataset):
    def __init__(self, dataset):
        self.label = dataset.labels.squeeze(-1).astype(int)
        self.feature = dataset.features
        #n_values = int(np.max(self.label) + 1)
        #self.label = np.eye(n_values)[self.label.astype(int)].squeeze(1)
        
    def __getitem__(self, idx):
    
        y = self.label[idx]
        x = self.feature[idx]
        
        return x, y
    
    
    def __len__(self):
        return len(self.label)
    

def fpr_calc(label, pred):
    FP = (sum(pred[(label == 0)])) #False Positive
    TN = (len(pred[(label == 0)]) - sum(pred[(label == 0)]))

    fpr = FP / (FP+TN)
    
    return fpr

def tpr_calc(label, pred):
    TP = (sum(pred[(label == 1)])) #False Positive
    FN = (len(pred[(label == 1)]) - sum(pred[(label == 1)]))

    tpr = TP / (TP+FN)
    
    return tpr