# Author: Yahui Liu <yahui.liu@unitn.it>

import os
import cv2
import numpy as np
import codecs
import glob

def imread(path, load_size=0, load_mode=cv2.IMREAD_GRAYSCALE, convert_rgb=False, thresh=-1):
    im = cv2.imread(path, load_mode)
    if convert_rgb:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    if load_size > 0:
        im = cv2.resize(im, (load_size, load_size), interpolation=cv2.INTER_CUBIC)
    if thresh > 0:
        _, im = cv2.threshold(im, thresh, 255, cv2.THRESH_BINARY)
    return im

def get_image_pairs(data_dir, suffix_gt='label_viz', suffix_pred='fused'):
    # print(data_dir,'111')./results/WDcascadeNet/CFD/0/test_latest/images 111
    gt_list = glob.glob(os.path.join(data_dir, '*{}.png'.format(suffix_gt)))
    # print(gt_list,'sss')[]
    pred_list = [ll.replace(suffix_gt, suffix_pred) for ll in gt_list]
    # print(pred_list,'ppp')[]
    
    assert len(gt_list) == len(pred_list)
    pred_imgs, gt_imgs = [], []
    for pred_path, gt_path in zip(pred_list, gt_list):
        pred=imread(pred_path)
        # print(pred.shape)
        
        gt=imread(gt_path, thresh=127)
        # print(gt.shape)
        if pred.shape[0]>pred.shape[1]:
            pred = np.rot90(pred, k=1).copy()
        if gt.shape[0]>gt.shape[1]:
            gt = np.rot90(gt, k=1).copy()
        # print(pred.shape)
        # print(gt.shape)

        pred_imgs.append(pred)
        gt_imgs.append(gt)
    return pred_imgs, gt_imgs

def save_results(input_list, output_path):
    with codecs.open(output_path, 'w', encoding='utf-8') as fout:
        for ll in input_list:
            line = '\t'.join(['%.4f'%v for v in ll])+'\n'
            fout.write(line)