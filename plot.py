import numpy as np
import matplotlib.pyplot as plt
from textwrap import wrap
import os
import math
import matplotlib

# methods_orig = ['Baseline', 'Adversarial Debiasing', 'Calibrated Equalized Odds Postprocessing',
# 'Disparate Impact Remover', 'Learning Fair Representations', 'Optimized Preprocessing', 'Reweighing',
# 'Reject Option Classification', 'Prejudice Remover Regularizer', 'FAIAS']

def plot(X_orig, metric_idx, data, setting, method_idx = [0,1,2,3,6,7,9]):
    methods_orig = ['Ours', 'AdvDeb', 'CEOPost', 'DIR', 'LFR', 'MFC',
                    'ReW', 'RejOpt', 'PRR', 'Ours', 'Baseline']
                                            
    metrics = ['Classification Accuracy', 'Balanced Classification Accuracy', 'Abs (1-Disparate Impact)',
               'Abs Equal Opportunity Difference', 'Abs Average Odds Difference', 'Abs Theil Index',
               'Abs Statistical Parity Difference', 'True Positive Rate', 'False Positive Rate Difference', 'Abs (1-Disp Imp norm)']

    # Enter raw data
    methods = ['\n'.join(wrap(methods_orig[i], 12)) for i in method_idx]
    # X_orig = np.random.rand(10, 5)
    X = X_orig[method_idx, :]

    # Create lists for the plot
    x_pos = np.arange(len(method_idx))
    # y_pos = np.arange(0.0,1.0,0.1)
    ymax = math.ceil(np.max(X)*10) / 10.0
    y_pos = np.arange(0.0, ymax, 0.1)

    # Build the plot
    fig, ax = plt.subplots()
    ax.bar(x_pos, np.mean(X, axis=1), yerr=np.std(X, axis=1), align='center', alpha=0.5, ecolor='black', capsize=10)
    ax.set_ylabel(metrics[metric_idx], fontsize=15)
    ax.set_xticks(x_pos)
    for axx in fig.axes:
        matplotlib.pyplot.sca(axx)
        plt.xticks(rotation=45)
    
    ax.set_xticklabels(methods, fontsize=11)
    # ax.set_yticks(y_pos)
    ax.set_ylim(ymin=0.0, ymax=ymax)
    ax.tick_params(axis='y', which='major', labelsize=15)
    # ax.tick_params(axis='x', which='major', labelsize=16)
#     ax.set_title(metrics[metric_idx] + ' for Dataset: ' + data, fontsize=14)
    ax.set_title(data, fontsize=14)
    ax.yaxis.grid(True)
    # ax.annotate('Fair',xy=(x_pos[-1]+.5,1.))
    # ax.annotate('Bias',xy=(x_pos[-1]+.5,-1.))

    if metric_idx in [2,3,4,5,6,8,9]:
        ax2 = ax.twinx()
        ax2.set_yticks(np.arange(2))
        ax2.set_yticklabels(['Fair', 'Bias'], fontsize=15, color='red')

    # Save the figure and show
    plt.tight_layout()
    if not os.path.exists('./plot_fig/'+setting):
        os.makedirs('./plot_fig/'+setting)

    plt.savefig('./plot_fig/'+setting+'/'+metrics[metric_idx].replace(' ', '_')+'_'+data+'.png')
    plt.show()
    # fig.savefig('test.png')