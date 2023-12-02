# train.py
#!/usr/bin/env	python3

""" valuate network using pytorch
    Junde Wu
"""

import os
import sys
import argparse
from datetime import datetime
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, accuracy_score,confusion_matrix
import torchvision
import torchvision.transforms as transforms
from skimage import io
from torch.utils.data import DataLoader
#from dataset import *
from torch.autograd import Variable
from PIL import Image
from tensorboardX import SummaryWriter
#from models.discriminatorlayer import discriminator
from dataset import *
from conf import settings
import time
import cfg
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from utils import *
import function


args = cfg.parse_args()

GPUdevice = torch.device('cuda', args.gpu_device)

net = get_network(args, args.net, use_gpu=args.gpu, gpu_device=GPUdevice, distribution = args.distributed)

'''load pretrained model'''
if args.pretrain:
    # checkpoint = torch.load('./logs/msa-3d-sam-btcv_2023_09_20_00_15_30/Model/checkpoint_best.pth')
    # checkpoint = torch.load('./logs/msa-3d-sam-btcv_2023_09_25_10_13_57/Model/checkpoint_best.pth')
    # checkpoint = torch.load('./logs/point_100/Model/checkpoint_best.pth')
    # checkpoint = torch.load('./logs/text_100/Model/checkpoint_best.pth')
    checkpoint = torch.load('./logs/point_transform_neg_small_overall/Model/checkpoint_best.pth')
    state_dict = checkpoint['state_dict']
    net.load_state_dict(state_dict)
    print('args.pretrain', args.pretrain)

# args.path_helper = checkpoint['path_helper']
# logger = create_logger(args.path_helper['log_path'])
# print(f'=> loaded checkpoint {checkpoint_file} (epoch {start_epoch})')

# args.path_helper = set_log_dir('logs', args.exp_name)
# logger = create_logger(args.path_helper['log_path'])
# logger.info(args)

# args.path_helper = set_log_dir('logs', args.exp_name)
# logger = create_logger(args.path_helper['log_path'])
# logger.info(args)


'''data loader & datapreprocess'''

# eval_dataloader = get_eval_dataset(args)
eval_dataloader = get_btcv_eval_dataset(args)

'''begain valuation'''
best_acc = 0.0
best_tol = 1e4
epoch = 1

# pred = torch.Tensor([[0,0,1,1], [0,1,0,1], [1,1,1,0], [1,1,0,1]])
# gt = torch.Tensor([[0,1,1,1], [0,0,0,1], [1,0,0,0], [1,0,0,0]])
# intersection, union = cal_intersection_union(pred, gt)
# print('check', intersection, union)
# organ_list = ['Liver', 'Lung Cancer', 'Pancreas', 'Hepatic Vessels', 'Spleen', 'Colon Cancer']

organ_list = [
    "spleen",
    "kidney right",
    "kidney left",
    "gallbladder",
    "esophagus",
    "liver",
    "stomach",
    "aorta",
    "inferior vena cava",
    "portal vein and splenic vein",
    "pancreas",
    "adrenal gland",
]

if args.mod == 'sam_adpt':
    net.eval()
    # organ_dice_list, organ_pos_dice_list, organ_neg_dice_list, organ_dice = function.eval_sam(args, eval_dataloader, epoch, net)
    organ_dice_list, organ_pos_dice_list, organ_neg_dice_list, organ_dice = function.eval_sam_btcv(args, eval_dataloader, epoch, net)
    for organ in organ_list:
        print(f'-------------------------{organ}------------------------')
        print(f'organ_dice : {sum(organ_dice_list[organ]) / len(organ_dice_list[organ])}, {sum(organ_dice[organ]) / len(organ_dice[organ])}.')
        print(f'pos_dice : {sum(organ_pos_dice_list[organ]) / len(organ_pos_dice_list[organ])}.')
        print(f'neg_dice : {sum(organ_neg_dice_list[organ]) / len(organ_neg_dice_list[organ])}.')
        print('check', len(organ_dice_list[organ]), len(organ_pos_dice_list[organ]), len(organ_neg_dice_list[organ]), len(organ_dice[organ]))


















# organ_dice_list, organ_pos_dice_list, organ_pos_pred_area_list, organ_neg_pred_area_list, organ_negative_correct_ratio_list, organ_positive_correct_ratio_list = function.eval_sam_test(args, eval_train_dataloader, epoch, net)
# print(f'Train || organ_dice_list : {organ_dice_list} organ_pos_dice_list : {organ_pos_dice_list} organ_pos_pred_area_list : {organ_pos_pred_area_list} organ_neg_pred_area_list : {organ_neg_pred_area_list} organ_negative_correct_ratio_list : {organ_negative_correct_ratio_list} organ_positive_correct_ratio_list : {organ_positive_correct_ratio_list} || @ epoch {epoch}.')
# val_organ_dice_list, val_organ_pos_dice_list, val_organ_pos_pred_area_list, val_organ_neg_pred_area_list, val_organ_negative_correct_ratio_list, val_organ_positive_correct_ratio_list = function.eval_sam_test(args, eval_valid_dataloader, epoch, net)
# print(f'Valid || organ_dice_list : {val_organ_dice_list} organ_pos_dice_list : {val_organ_pos_dice_list} organ_pos_pred_area_list : {val_organ_pos_pred_area_list} organ_neg_pred_area_list : {val_organ_neg_pred_area_list} organ_negative_correct_ratio_list {val_organ_negative_correct_ratio_list} organ_positive_correct_ratio_list : {val_organ_positive_correct_ratio_list} || @ epoch {epoch}.')
