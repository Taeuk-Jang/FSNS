import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


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
    
class SVM(nn.Module):
    def __init__(self, input_size, num_cls = 1):
        super(SVM, self).__init__()      
        self.dense1 = nn.Linear(input_size, num_cls)
        
    def forward(self, x):
        return self.dense1(x)
    
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
        
        x = F.softmax(self.dense6(x), dim= -1)
        
        return x 
    
def init_weights(m):
    print(m)
    if type(m) == nn.Linear:
        torch.nn.init.kaiming_normal_(m.weight)