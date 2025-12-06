# Author: Yahui Liu <yahui.liu@unitn.it>

"""
Calculate sensitivity and specificity metrics:
 - Precision
 - Recall
 - F-score
"""

import numpy as np
from data_io import imread
import matplotlib.pyplot  as plt
import matplotlib.font_manager as fm
import os

font_path = './front/Times New Roman.ttf'  # 替换为 Times New Roman 字体的实际路径
font_prop = fm.FontProperties(fname=font_path)

def cal_prf_metrics(pred_list, gt_list, thresh_step=0.01,metric_path='',dataset=''):
    final_accuracy_all = []
    precision,recall=[],[]
    f1_score=[]
    tpr,fpr=[],[]
    
    for thresh in np.arange(0.0, 1.0, thresh_step):
        # print(thresh)
        statistics = []
        
        for pred, gt in zip(pred_list, gt_list):
            # print( pred.shape != gt.shape)
            gt_img   = (gt/255).astype('uint8')
            pred_img = (pred/255 > thresh).astype('uint8')
            # calculate each image
            statistics.append(get_statistics(pred_img, gt_img))
        
        # get tp, fp, fn
        tp = np.sum([v[0] for v in statistics])
        fp = np.sum([v[1] for v in statistics])
        fn = np.sum([v[2] for v in statistics])
        tn = np.sum([v[3] for v in statistics])
        tprate=tp/(tp+fn)
        fprate=fp/(fp+tn)
        tpr.append(tprate)
        fpr.append(fprate)
        # calculate precision
        p_acc = 1.0 if tp==0 and fp==0 else tp/(tp+fp)
        # p_acc = tp/(tp+fp)
        # calculate recall
        r_acc = tp/(tp+fn) if (tp + fn) != 0 else 0
        # r_acc = tp/(tp+fn)
        # calculate f-score
        f1=2*p_acc*r_acc/(p_acc+r_acc) if (p_acc + r_acc) != 0 else 0

        precision.append(p_acc)
        recall.append(r_acc)
        f1_score.append(f1)

        final_accuracy_all.append([thresh, p_acc, r_acc, f1])
    max_f1 = np.max(f1_score)
    # print(max_f1) 0.5731832999053414

    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    plt.plot(recall, precision, color='blue', lw=2, label=dataset+f' (F1 = {max_f1:.2f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontproperties=font_prop)
    plt.ylabel('Precision', fontproperties=font_prop)
    plt.xticks(fontproperties=font_prop)
    plt.yticks(fontproperties=font_prop)
    plt.title('Precision-Recall curve', fontproperties=font_prop)
    plt.legend(loc='lower left',prop=font_prop)

    plt.subplot(1, 2, 2)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontproperties=font_prop)
    plt.ylabel('True Positive Rate', fontproperties=font_prop)
    plt.xticks(fontproperties=font_prop)
    plt.yticks(fontproperties=font_prop)
    plt.title('Receiver Operating Characteristic', fontproperties=font_prop)
    plt.legend(loc='lower right', prop=font_prop)

    plt.subplots_adjust(wspace=0.3)
    os.makedirs(metric_path, exist_ok=True)
    plt.savefig(metric_path+'/PR&ROC.png')

    return final_accuracy_all

def get_statistics(pred, gt):
    """
    return tp, fp, fn
    """
    tp = np.sum((pred==1)&(gt==1))
    fp = np.sum((pred==1)&(gt==0))
    fn = np.sum((pred==0)&(gt==1))
    tn = np.sum((pred==0)&(gt==0))

    return [tp, fp, fn, tn] 
