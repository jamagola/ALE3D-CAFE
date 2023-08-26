import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

pred_npy=np.load('pd.npy')
truth_npy=np.load('gt.npy')

def plot_confusion_matrix(df_confusion, title='Confusion norm matrix', cmap=plt.cm.hot):
    plt.matshow(df_confusion, cmap=cmap) # imshow
    #plt.title(title)
    plt.colorbar()
    tick_marks_x = np.arange(len(df_confusion.columns))
    tick_marks_y = np.arange(len(df_confusion.index))
    plt.xticks(tick_marks_x, df_confusion.columns, rotation=45)
    plt.yticks(tick_marks_y, df_confusion.index)
    plt.tight_layout()
    plt.ylabel(df_confusion.index.name)
    plt.xlabel(df_confusion.columns.name)
    plt.show()

def confusionAll(truth_npy, pred_npy, step=0, feature=0):
    pred_npy[...,feature]=np.round(pred_npy[...,feature]).astype(int)
    truth_npy[...,feature]=np.round(truth_npy[...,feature]).astype(int)

    y_actu=truth_npy[:,step,:,:,:,feature].reshape(-1)
    y_pred=pred_npy[:,step,:,:,:,feature].reshape(-1)

    df_confusion = pd.crosstab(y_actu, y_pred, rownames=['Actual'], colnames=['Predicted'], margins=False)
    df_conf_norm = df_confusion / df_confusion.sum(axis=1)

    print('Confusion Matrix: ')
    print(df_confusion)
    plot_confusion_matrix(df_conf_norm)


def confusionFeat(truth_npy, pred_npy, step=0, feature=0, cellid=0):
    pred_npy[...,feature]=np.round(pred_npy[...,feature]).astype(int)
    truth_npy[...,feature]=np.round(truth_npy[...,feature]).astype(int)

    y_actu=np.array([])
    y_pred=np.array([])
    # Data loading: valid no shuffle and clip step=1
    for seq in np.arange(truth_npy.shape[0]-1):
        index=truth_npy[seq,step,:,:,:,feature]==cellid # 0,1,2,3 cell state
        temp = truth_npy[seq+1,step,index,feature]
        y_actu=np.append(y_actu, temp)
        temp = pred_npy[seq+1,step,index,feature]
        y_pred=np.append(y_pred, temp)

    df_confusion = pd.crosstab(y_actu, y_pred, rownames=['Actual'], colnames=['Predicted'], margins=False)
    df_conf_norm = df_confusion / df_confusion.sum(axis=1)

    print('Confusion Matrix: ')
    print(df_confusion)
    plot_confusion_matrix(df_conf_norm)

confusionFeat(truth_npy, pred_npy, step=0, feature=0, cellid=0)
#confusionAll(truth_npy, pred_npy, step=0, feature=0)
