import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import sklearn.metrics as sklm

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import scale, StandardScaler, MaxAbsScaler
from sklearn.metrics import accuracy_score


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


class Classifier(nn.Module):
    def __init__(self, input_size, output_size=2):
        super(Classifier, self).__init__()
        
        self.dense1 = nn.Linear(input_size, 1024)
        #self.bn1 = nn.BatchNorm1d(1024)
        
        self.dense2 = nn.Linear(1024, 1024)
        #self.bn2 = nn.BatchNorm1d(1024)        
        
        self.dense3 = nn.Linear(1024, 1024)
        #self.bn3 = nn.BatchNorm1d(1024)  
        
        self.dense4 = nn.Linear(1024, 1024)
        #self.bn4 = nn.BatchNorm1d(1024)    
        
        #self.do = nn.Dropout(0.3)
        
        self.dense5 = nn.Linear(1024, output_size)
        
    def forward(self, x):
        x = F.relu(self.dense1(x))
        #x = self.bn1(x)
        #x = self.do(x)
        
        x = F.relu(self.dense2(x))
        #x = self.bn2(x)
        #x = self.do(x)
        
        x = F.relu(self.dense3(x))
        #x = self.bn3(x)
        #x = self.do(x)
                
        x = F.relu(self.dense4(x))
        #x = self.bn4(x)
        #x = self.do(x)
 
        x = (self.dense5(x))
        
        return x        
    
    
class Feature_extractor(nn.Module):
    def __init__(self, input_size, output_size = 50, latent_size = 256):
        super(Feature_extractor, self).__init__()
        
        self.dense1 = nn.Linear(input_size, latent_size)
        
        self.dense2 = nn.Linear(latent_size, latent_size)       
        
        self.dense3 = nn.Linear(latent_size, output_size)
        
    def forward(self, x):
        x = F.relu(self.dense1(x))

        x = F.relu(self.dense2(x))

        x = self.dense3(x)
        
        return x   
    
class Generator(nn.Module):
    def __init__(self, input_size, output_size, latent_size = 1024):
        super(Generator, self).__init__()
        
        self.dense1 = nn.Linear(input_size + 2, latent_size)
    
        self.dense2 = nn.Linear(latent_size, latent_size)
     
        self.dense3 = nn.Linear(latent_size, latent_size)
   
        self.dense4 = nn.Linear(latent_size, latent_size)   
        
        self.dense5 = nn.Linear(latent_size, output_size + 1)
        
    def forward(self, x, a, y):
        x = torch.cat((x, a, y), 1)
        
        x = F.relu(self.dense1(x))

        x = F.relu(self.dense2(x))

        x = F.relu(self.dense3(x))

        x = F.relu(self.dense4(x))        
        
        x = F.relu(self.dense5(x))

        x_g, y_g = x[:, :-1], x[:, -1]
        
        y_g[y_g > 0.5] = 1.0
        y_g[y_g <= 0.5] = 0.0
        
        return x_g, y_g
    
class Discriminator(nn.Module):
    def __init__(self, input_size, latent_size = 1024):
        super(Discriminator, self).__init__()
        
        self.dense1 = nn.Linear(input_size + 1, latent_size)
    
        self.dense2 = nn.Linear(latent_size, latent_size)
     
        self.dense3 = nn.Linear(latent_size, latent_size)
   
        self.dense4 = nn.Linear(latent_size, latent_size)   
        
        self.dense5 = nn.Linear(latent_size, 1)
        
    def forward(self, x, y):
        x = torch.cat((x, y), 1)
        
        x = F.relu(self.dense1(x))

        x = F.relu(self.dense2(x))

        x = F.relu(self.dense3(x))

        x = F.relu(self.dense4(x))        
        
        x = F.sigmoid(self.dense5(x))
        
        return x
    
class Network():
    def __init__(self, input_size, latent_size, device, epochs, lr):
        self.H = Feature_extractor(input_size, latent_size).double().to(device)
        self.G = Generator(latent_size, input_size).double().to(device)
        self.D = Discriminator(input_size).double().to(device)
        self.C = Classifier(input_size).double().to(device)
        
        self.H.apply(init_weights)
        self.D.apply(init_weights)
        self.G.apply(init_weights)
        self.C.apply(init_weights)
        
        self.optimizerD = optim.Adam(self.D.parameters(), lr=lr[1], betas=(0.5, 0.999))
        self.optimizerG = optim.Adam(self.G.parameters(), lr=lr[1], betas=(0.5, 0.999))
        self.optimizerH = optim.Adam(self.H.parameters(), lr=lr[0])
        self.optimizerC = optim.Adam(self.C.parameters(), lr=lr[2])
        self.celoss = nn.functional.cross_entropy
        self.mseloss = nn.functional.mse_loss
        self.criterion = nn.BCELoss()

        self.device = device
        self.epochs = epochs
        
    def train(self, trainloader, validloader):
        for epoch in range(1, self.epochs+1):
            acc, acc_fake = 0, 0
            cnt = 0
            for x, a, y in trainloader:
                cnt += 1
                x, a, y = x.to(self.device), a.to(self.device), y.to(self.device)
                
                ##############################
                ####### train extractor ######
                ##############################
                self.optimizerH.zero_grad()
                
                h = self.H(x)
                a_flip = abs(a-3)
                
                x_g, y_g = self.G(h.detach(), a_flip, y.double().unsqueeze(1))
                h_g = self.H(x_g.detach())
                
                loss_H = self.mseloss(h_g, h)
                
                loss_H.backward()
                self.optimizerH.step()
                
                ##############################
                ##### train discriminator ####
                ##############################
                self.optimizerD.zero_grad()
                ### Train with real ones ###
                pred_real = self.D(x, y.double().unsqueeze(1))
                
                label = torch.full((x.shape[0],), 1, device=self.device).double()
                
                loss_D_real = self.criterion(pred_real, label)
                loss_D_real.backward()
                
                D_x = pred_real.mean().item()
                
                ### Train with fake ones ###
                pred_fake = self.D(x_g.detach(), y.unsqueeze(1).detach())
                
                label.fill_(0)
                
                loss_D_fake = self.criterion(pred_fake, label)
                loss_D_fake.backward()
                
                loss_D = loss_D_fake + loss_D_real
                
                self.optimizerD.step()
                
                D_G_z1 = pred_fake.mean().item()
                                
                ##############################
                ####### train Generator ######
                ##############################
                self.optimizerG.zero_grad()
                
                label.fill_(1)
                
                pred_fake = self.D(x_g, y.view(-1,1))
                
                loss_G = self.criterion(pred_fake, label)
                loss_G.backward()
                
                self.optimizerG.step()
                
                D_G_z2 = pred_fake.mean().item()
                
                ##############################
                ####### train Classifier #####
                ##############################
                self.optimizerC.zero_grad()
                
                cls_fake = self.C(x_g.detach())
                
                cls_real = self.C(x)
                
                loss_C_real = self.celoss(cls_real, y)
                loss_C_fake = self.celoss(cls_fake, y)
                loss_fair = self.mseloss(cls_fake, cls_real)
                
                loss_C = loss_C_real + loss_C_fake + loss_fair
                
                loss_C.backward()
                
                self.optimizerC.step()
                
                acc += sum(cls_real.argmax(-1) == y)/float(len(y))
                acc_fake += sum(cls_fake.argmax(-1) == y)/float(len(y))                
                
            acc /= cnt
            acc_fake /= cnt
            
            print('epoch : {}\n\n'.format(epoch))
            print('loss_H : {:.6f}'.format(loss_H.item()))
            print('loss_G : {:.5f}, \tloss_D : {:.5f}'.format(loss_G.item(), loss_D.item() ))
            print('D(x): {:.4f}, \tD(G(z)): {:.5f} / {:.5f}'.format(D_x, D_G_z1, D_G_z2))
            print('loss_C : {:.4f}'.format(loss_C.item()))
                        
            print('ACC   real : {:.3f}, \tfake : {:.3f}\n\n'.format(acc, acc_fake))
            
            if epoch % 25 == 0:
                tpr_overall, tpr_priv, tpr_unpriv,\
                fpr_overall, fpr_unpriv, fpr_priv,\
                acc_overall, acc_priv, acc_unpriv, eq_overall = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                cnt = 0
                
                for x, a, y in validloader:
                    cnt += 1
                    x, a, y = x.to(self.device), a.to(self.device), y.to(self.device)
                    test_priv = x[(a == 2.).view(-1)]
                    test_unpriv = x[(a == 1.).view(-1)]

                    test_lb_priv = y[(a == 2.).view(-1)]
                    test_lb_unpriv = y[(a == 1.).view(-1)]

                    pred_test = self.C(x)
                    pred_priv = self.C(test_priv)
                    pred_unpriv = self.C(test_unpriv)

                    y = y.cpu().detach().numpy()
                    test_lb_priv = test_lb_priv.cpu().detach().numpy()
                    test_lb_unpriv = test_lb_unpriv.cpu().detach().numpy()


                    pred_test = pred_test.cpu().detach().numpy().argmax(1)
                    pred_priv = pred_priv.cpu().detach().numpy().argmax(1)
                    pred_unpriv = pred_unpriv.cpu().detach().numpy().argmax(1)

                    tpr_overall += sklm.recall_score(y, pred_test) #act recidivism, detection rate.
                    tpr_priv += sklm.recall_score(test_lb_priv, pred_priv) #act recidivism, detection rate.
                    tpr_unpriv += sklm.recall_score(test_lb_unpriv, pred_unpriv) #act recidivism, detection rate.

                    fpr_overall += fpr_calc(y, pred_test) #act recidivism, detection rate.
                    fpr_priv += fpr_calc(test_lb_priv, pred_priv) #act recidivism, detection rate.
                    fpr_unpriv += fpr_calc(test_lb_unpriv, pred_unpriv) #act recidivism, detection rate.

                    acc_overall += sklm.accuracy_score(y, pred_test)
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
                    
                print('\n epoch {}'.format(epoch))
                print()
                print('overall TPR : {0:.3f}'.format( tpr_overall))
                print('priv TPR : {0:.3f}'.format( tpr_priv))
                print('unpriv TPR : {0:.3f}'.format( tpr_unpriv))
                print('Eq. Opp : {0:.3f}'.format( abs(tpr_unpriv - tpr_priv)))
                print()
                print('overall FPR : {0:.3f}'.format( fpr_overall))
                print('priv FPR : {0:.3f}'.format( fpr_priv))
                print('unpriv FPR : {0:.3f}'.format( fpr_unpriv))
                print('diff FPR : {0:.3f}'.format( abs(fpr_unpriv-fpr_priv)))
                print()
                print('overall ACC : {0:.3f}'.format( acc_overall))
                print('priv ACC : {0:.3f}'.format( acc_priv))
                print('unpriv ACC : {0:.3f}'.format( acc_unpriv)) 
                print('diff ACC : {0:.3f}\n\n\n'.format( abs(acc_unpriv-acc_priv)))

def init_weights(m):
    print(m)
    if type(m) == nn.Linear:
        torch.nn.init.kaiming_normal_(m.weight)
        
        

