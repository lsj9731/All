import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import pandas as pd
import scipy
import os
import dataloader
import time
import model
from opts import parser
from tqdm import tqdm
import tensorflow_hub as hub
import datetime
import custom_callbacks
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score

def main():
    args = parser.parse_args()

    # PhysioNet2012 goal is binary classification
    num_class = 1
    test_dataset = dataloader.Dataloader(args.dirs, args.batch, 'Testing', 'PhysioNet2012')

    # output_activation, output_dims, n_dims, n_heads, n_layers, dropout, attn_dropout, aggregation_fn, max_timescale):
    print('Construct model')
    Transformer = model.TransformerModel('sigmoid', num_class, args.dims, args.heads, args.layers, args.dropout, args.dropout, 'mean', 100.0)
    print('Done.')

    # ex)  --checkpoint best_weights/41epochs/Model_Best_Checkpoint_41_epochs
    if args.checkpoint is not None:
        print('Get saved model weights !')
        print('We will resume model training.')
        Transformer.load_weights(args.checkpoint)
    elif args.checkpoint is None:
        raise Exception('Please enter the path.')
    
    probabilitys, results, labels = [], [], []
    for datas, label in test_dataset:
        predictions = Transformer(datas, training=False)
        probabilitys.extend(predictions)
        results.extend(predictions)
        labels.extend(label)

    # Find optimal probability threshold
    threshold = Find_Optimal_Cutoff(labels, probabilitys)
    print('cut-off value : ', threshold)

    # Categorize by threshold
    ones_zeros = []
    thresholds = float(threshold[0]) + 0.05
    for p in results:
        if p >= thresholds:
            ones_zeros.append(1)
        elif p < thresholds:
            ones_zeros.append(0)

    # Accuracy, recall, precision, f1 score
    get_clf_eval(labels, ones_zeros)


    # AUROC curve score
    auc_score = roc_auc_score(labels, probabilitys)
    print('AUROC 점수 : ', auc_score)

    
def get_clf_eval(y_test, pred):
    confusion = confusion_matrix(y_test, pred)
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    f1score = f1_score(y_test, pred)
    print('Confusion Matrix\n')
    print(confusion, '\n')
    print('정확도 : {}, 정밀도 : {}, 재현율 : {}, 조화평균 : {}'.format(accuracy, precision, recall, f1score))


def Find_Optimal_Cutoff(target, predicted):
    """ Find the optimal probability cutoff point for a classification model related to event rate
    Parameters
    ----------
    target : Matrix with dependent or target data, where rows are observations

    predicted : Matrix with predicted data, where rows are observations

    Returns
    -------     
    list type, with optimal cutoff value
        
    """
    fpr, tpr, threshold = roc_curve(target, predicted)
    i = np.arange(len(tpr)) 
    roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i)})
    roc_t = roc.iloc[(roc.tf-0).abs().argsort()[:1]]

    return list(roc_t['threshold']) 


if __name__ == "__main__":
    main()