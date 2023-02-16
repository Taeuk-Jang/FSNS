import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.preprocessing import scale, StandardScaler, MaxAbsScaler
from utils import *
import sys
sys.path.append("../")
from aif360.metrics import ClassificationMetric

celoss = torch.nn.BCELoss()
min_max_scaler = MaxAbsScaler()

def evaluate(args, repeat, epoch, dataset, dataloader, H, P, C, sens_idx, num_sens, privileged_groups, unprivileged_groups,  device, test):
    VIEW_EVAL = False
    tpr_overall, tpr_priv, tpr_unpriv,\
    fpr_overall, fpr_unpriv, fpr_priv,\
    acc_overall, acc_priv, acc_unpriv, eq_overall = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    cnt = 0

    tp_priv, tn_priv, fp_priv, fn_priv, \
    tp_unpriv, tn_unpriv, fp_unpriv, fn_unpriv = 0, 0, 0, 0, 0, 0, 0, 0

    with torch.no_grad():
        for x_test, a_test, y_test in dataloader:
            cnt += 1

            x_test, a_test, y_test = x_test.to(device), a_test.to(device), y_test.to(device)
            a_onehot = one_hot_embedding(a_test, num_sens).to(device)
            y_onehot = one_hot_embedding(y_test, 2).to(device)
            a_r = torch.zeros_like(a_onehot).float()
            a_r += 1 / num_sens #uniform dist

            priv_idx = (a_test==1).squeeze()
            positive_idx = y_test==1

            latent_test = H(x_test)
            sens_test = P(latent_test)
            pred_test = C(latent_test)

            h_priv = H(x_test[priv_idx])
            h_unpriv = H(x_test[~priv_idx])

            h_positive = H(x_test[positive_idx])

            loss_C = svm_loss(y_onehot, pred_test)
            loss_P = celoss(sens_test[:,1], a_test.double())


            cly_loss = max_margin_loss(y_onehot, pred_test, args.alpha, args.mu, args.lamda, device)
            ar_loss = cross_entropy(a_r, sens_test)

            loss_H = cly_loss + ar_loss                      


            test_lb_priv = y_test[priv_idx]
            test_lb_unpriv = y_test[~priv_idx]

            pred_priv = C(h_priv)
            pred_unpriv = C(h_unpriv)

            y_test = y_test.cpu().detach().numpy()
            test_lb_priv = test_lb_priv.cpu().detach().numpy()
            test_lb_unpriv = test_lb_unpriv.cpu().detach().numpy()

            try:
                pred_priv = pred_priv.argmax(1)
            except:
                pass
            try:
                pred_unpriv = pred_unpriv.argmax(1)
            except:
                pass


            tp_priv += sum(pred_priv[test_lb_priv == 1] == 1)
            fp_priv += sum(pred_priv[test_lb_priv == 0] == 1)
            tn_priv += sum(pred_priv[test_lb_priv == 0] == 0)
            fn_priv += sum(pred_priv[test_lb_priv == 1] == 0)

            tp_unpriv += sum(pred_unpriv[test_lb_unpriv == 1] == 1)
            fp_unpriv += sum(pred_unpriv[test_lb_unpriv == 0] == 1)
            tn_unpriv += sum(pred_unpriv[test_lb_unpriv == 0] == 0)
            fn_unpriv += sum(pred_unpriv[test_lb_unpriv == 1] == 0)

        tpr_overall = (tp_priv + tp_unpriv)/(tp_priv + tp_unpriv + fn_priv + fn_unpriv).float().item()
        tpr_unpriv = (tp_unpriv)/(tp_unpriv + fn_unpriv).float().item()
        tpr_priv = (tp_priv)/(tp_priv + fn_priv).float().item()

        fpr_overall = (fp_priv + fp_unpriv)/(tn_priv + tn_unpriv + fp_priv + fp_unpriv).float().item()
        fpr_unpriv = (fp_unpriv)/(tn_unpriv + fp_unpriv).float().item()
        fpr_priv = (fp_priv)/(tn_priv + fp_priv).float().item()

        acc_overall = (tp_priv + tn_priv + tp_unpriv + tn_unpriv)/(tp_priv + tn_priv + tp_unpriv + tn_unpriv + \
                                                                  fp_priv + fn_priv + fp_unpriv + fn_unpriv).float().item()
        acc_priv = (tp_priv + tn_priv)/(tp_priv + tn_priv + fp_priv + fn_priv).float().item()
        acc_unpriv = (tp_unpriv + tn_unpriv)/(tp_unpriv + tn_unpriv + fp_unpriv + fn_unpriv).float().item()

        if test:
            print('\n TEST {}-{}'.format(repeat, epoch))
        else:
            print('\n VALID {}-{}'.format(repeat, epoch))
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

        test_pred = dataset.copy(deepcopy=True)
        feature_size = test_pred.features.shape[1]
        sens_loc = np.zeros(feature_size).astype(bool)
        sens_loc[sens_idx] = 1

        feature = test_pred.features[:,~sens_loc] #data without sensitive
        feature = min_max_scaler.fit_transform(feature)

        test_pred.labels = C(H(torch.tensor(feature).to(device))).argmax(-1).cpu().numpy()

        classified_metric = ClassificationMetric(dataset,
                                                         test_pred,
                                                         unprivileged_groups=unprivileged_groups,
                                                         privileged_groups=privileged_groups)


        print('balanced acc :' ,1/2*(classified_metric.true_positive_rate() + classified_metric.true_negative_rate()))
        print('disparate_impact :' ,classified_metric.disparate_impact())
        print('theil_index :' ,classified_metric.theil_index())
        print('statistical_parity_difference :' ,classified_metric.statistical_parity_difference())
        
        lst  = [tpr_overall.item(), tpr_priv.item(), tpr_unpriv.item(), abs(tpr_unpriv.item() - tpr_priv.item()), fpr_overall.item(), fpr_priv.item(), fpr_unpriv.item(), abs(fpr_unpriv.item()-fpr_priv.item()), acc_overall.item(), acc_priv.item(), acc_unpriv.item(), abs(acc_unpriv.item()-acc_priv.item()), 1/2*(classified_metric.true_positive_rate() + classified_metric.true_negative_rate()), classified_metric.disparate_impact(), classified_metric.theil_index(), classified_metric.statistical_parity_difference()]
        
        if test:          
            return lst
        else:
            if (tpr_overall>0.65 and acc_overall > 0.65 and abs(acc_priv -  acc_unpriv) < 0.1).item():
                VIEW_EVAL = True
            return VIEW_EVAL
        

def vis(latent, a, y):
    latent = latent.cpu().detach().numpy()
    latent = StandardScaler().fit_transform(latent) # normalizing the features

    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(latent)


    fig = plt.figure()
    plt.scatter(principalComponents[a.cpu().detach().numpy().squeeze()==0][:, 0],\
        principalComponents[a.cpu().detach().numpy().squeeze()==0][:, 1], marker='.')
    plt.scatter(principalComponents[a.cpu().detach().numpy().squeeze()==1][:,0],\
            principalComponents[a.cpu().detach().numpy().squeeze()==1][:,1], marker='.')
    plt.legend(['unpriv', 'priv'])
    plt.title('PCA_sens')
    plt.show()

    #writerid.add_figure('PCA_sens', fig, epoch)

    fig = plt.figure()
    plt.scatter(principalComponents[y.squeeze()==1][:, 0],\
                principalComponents[y.squeeze()==1][:, 1], marker='.')
    plt.scatter(principalComponents[y.squeeze()==0][:, 0],\
                principalComponents[y.squeeze()==0][:, 1], marker='.')
    plt.legend(['positive', 'negative'])
    plt.title('PCA_label')
    plt.show()

    #writerid.add_figure('PCA_label', fig, epoch)

    latent = H(x)

    embedded = tsne.fit_transform(latent.cpu().detach())

    vis_x = embedded[:, 0]
    vis_y = embedded[:, 1]

    fig = plt.figure()
    plt.scatter(vis_x, vis_y, c=a.cpu()*8, cmap=plt.cm.get_cmap("jet", 10), marker='.')
    #plt.colorbar(ticks=range(10))
    plt.clim(-0.5, 9.5)
    plt.title('Tsne_sens')
    plt.show()
    #writerid.add_figure('Tsne_sens', fig, epoch)

    fig = plt.figure()
    plt.scatter(vis_x, vis_y, c=y*8, cmap=plt.cm.get_cmap("jet", 10), marker='.')
    #plt.colorbar(ticks=range(10))
    plt.clim(-0.5, 9.5)
    plt.title('Tsne_label')
    plt.show()
    #writerid.add_figure('Tsne_label', fig, epoch)


    latent = H(x)
    latent = latent.cpu().detach().numpy()
    latent = StandardScaler().fit_transform(latent) # normalizing the features


    fig = plt.figure()
    plt.scatter(principalComponents[y.cpu().detach().numpy().squeeze()==1][:, 0],\
                principalComponents[y.cpu().detach().numpy().squeeze()==1][:, 1], marker='o', c= 'b', s= 20)
    plt.scatter(principalComponents[y.cpu().detach().numpy().squeeze()==-1][:, 0],\
                principalComponents[y.cpu().detach().numpy().squeeze()==-1][:, 1], marker='o', c= 'r', s= 20)

    principalComponents = latent.dot(pca.components_.T)

    plt.scatter(principalComponents[y.squeeze()==1][:, 0],\
                principalComponents[y.squeeze()==1][:, 1], marker='x', c= 'b', s= 20)
    plt.scatter(principalComponents[y.squeeze()==0][:, 0],\
                principalComponents[y.squeeze()==0][:, 1], marker='x', c= 'r', s= 20)
    plt.legend(['positive', 'negative', 'positive', 'negative'])
    plt.title('PCA_label')
    plt.show()
    #writerid.add_figure('PCA_label on train plane', fig, epoch)


    fig = plt.figure()
    plt.scatter(principalComponents[a.cpu().detach().numpy().squeeze()==1][:, 0],\
                principalComponents[a.cpu().detach().numpy().squeeze()==1][:, 1], marker='o', c= 'b', s= 15)
    plt.scatter(principalComponents[a.cpu().detach().numpy().squeeze()==0][:, 0],\
                principalComponents[a.cpu().detach().numpy().squeeze()==0][:, 1], marker='o', c= 'r', s= 15)

    plt.scatter(principalComponents[a.cpu().detach().numpy().squeeze()==1][:, 0],\
                principalComponents[a.cpu().detach().numpy().squeeze()==1][:, 1], marker='x', c= 'b', s= 20)
    plt.scatter(principalComponents[a.cpu().detach().numpy().squeeze()==0][:, 0],\
                principalComponents[a.cpu().detach().numpy().squeeze()==0][:, 1], marker='x', c= 'r', s= 20)
    plt.legend(['priv', 'unpriv', 'priv', 'unpriv'])
    plt.title('PCA_sens')
    plt.show()