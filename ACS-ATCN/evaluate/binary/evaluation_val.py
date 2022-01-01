import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
from sklearn.metrics import confusion_matrix,multilabel_confusion_matrix


def WeightedCal(weightli,valli):
    weigvalsum = 0
    all = sum(weightli)
    for i in range(len(valli)):
        val = valli[i]
        weight = weightli[i]/all
        weightval = weight*val
        weigvalsum = weigvalsum+weightval

    return weigvalsum


def Evaluate(GroundTruth,Prediction):
    GroundTruth_ = GroundTruth.copy()
    if GroundTruth_.shape[1]>2:
        GroundTruth_Idx = np.argmax(GroundTruth, axis=1).tolist()
        Prediction_Idx = np.argmax(Prediction, axis=1).tolist()

        conf_matx = multilabel_confusion_matrix(GroundTruth_Idx, Prediction_Idx)
        Gmeanli = []
        fscoreli = []
        candatenum = []
        for i in range(conf_matx.shape[0]):
            mtrix = conf_matx[i,:,:]
            tn, fp, fn, tp = mtrix.ravel()
            Precision = tp / (tp + fp)
            Recall = tp / (tp + fn)
            F1_score = 2 * Precision * Recall / (Precision + Recall)
            specificity = tn / (tn + fp)
            G_mean = (Recall * specificity) ** 0.5
            classnum = fn + tp
            Gmeanli.append(G_mean)
            fscoreli.append(F1_score)
            candatenum.append(classnum)

        G_mean_weight = WeightedCal(candatenum, Gmeanli)
    else:
        GroundTruth_Idx = np.argmax(GroundTruth, axis=1).tolist()
        Prediction_Idx = np.argmax(Prediction, axis=1).tolist()

        conf_matx = confusion_matrix(GroundTruth_Idx, Prediction_Idx)
        tn, fp, fn, tp = conf_matx.ravel()
        Recall = tp / (tp + fn)
        specificity = tn / (tn + fp)
        G_mean = (Recall * specificity) ** 0.5

    if GroundTruth_.shape[1] > 2:
        return G_mean_weight
    else:
        return G_mean
