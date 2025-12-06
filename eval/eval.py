# Author: Yahui Liu <yahui.liu@unitn.it>

import os
import numpy as np
import data_io
from prf_metrics import cal_prf_metrics
from segment_metrics import cal_semantic_metrics
import argparse
# import torch
# print(torch.__version__)2.5.1


parser = argparse.ArgumentParser()
parser.add_argument('--metric_mode', type=str, default='sem', help='[prf | sem]')
parser.add_argument('--model_name', type=str, default='WDcascadeNet')
parser.add_argument('--results_dir', type=str, default='./results')
parser.add_argument('--dataset_dir', type=str, default='CRACK500')
parser.add_argument('--metric_dir', type=str, default='./metric_pic')
parser.add_argument('--suffix_gt', type=str, default='label_viz', help='Suffix of ground-truth file name')
parser.add_argument('--suffix_pred', type=str, default='fused', help='Suffix of predicted file name')
parser.add_argument('--output_prf', type=str, default='prf-result.txt')
parser.add_argument('--output_sematic', type=str, default='semantic-result.txt')
parser.add_argument('--thresh_step', type=float, default=0.01)
parser.add_argument('--gpu_ids', type=str, default='1', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
args = parser.parse_args()

if __name__ == '__main__':
    metric_mode = args.metric_mode
    results_dir = os.path.join(args.results_dir, args.model_name,args.dataset_dir,args.gpu_ids, 'test_latest', 'images')
    # print(results_dir)
    # ./results/WDcascadeNet/CFD/0/test_latest/images
    # print(args.suffix_gt)
    src_img_list, tgt_img_list = data_io.get_image_pairs(results_dir, args.suffix_gt, args.suffix_pred)
    # 得到的是包含真实和预测的图像列表
    final_results = []
    # print(len(src_img_list))0
    # 118
    metric_path=args.metric_dir+'/'+args.model_name+'/'+args.dataset_dir+'/'+args.gpu_ids+'/'
    
    if metric_mode == 'prf':
        final_results = cal_prf_metrics(src_img_list,tgt_img_list, args.thresh_step,metric_path,args.dataset_dir)
        data_io.save_results(final_results, metric_path+args.model_name+'_'+args.output_prf)
        print("Results saved.")
    elif metric_mode == 'sem':
        final_results = cal_semantic_metrics(src_img_list, tgt_img_list, args.thresh_step)
        data_io.save_results(final_results, metric_path+args.model_name+'_'+args.output_sematic)
        print("Results saved.")
    else:
        print("Unknown mode of metrics.")

    

