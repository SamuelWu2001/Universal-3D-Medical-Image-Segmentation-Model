# train.py
#!/usr/bin/env	python3

""" train network using pytorch
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

seed = 513
torch.manual_seed(seed)  # 设置PyTorch的随机种子
torch.cuda.manual_seed(seed)  # 设置PyTorch的CUDA随机种子
torch.cuda.manual_seed_all(seed)  # 设置多个GPU的随机种子
random.seed(seed)  # 设置Python内置的random模块的随机种子
np.random.seed(seed)  # 设置NumPy的随机种子

args = cfg.parse_args()

GPUdevice = torch.device('cuda', args.gpu_device)

net = get_network(args, args.net, use_gpu=args.gpu, gpu_device=GPUdevice, distribution = args.distributed)
# if args.pretrain:
#     weights = torch.load(args.pretrain)
#     net.load_state_dict(weights,strict=False)

freeze_model(net)
# check_freeze_status(net)

optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5) #learning rate decay


starting_epoch = 0
'''load pretrained model'''
if args.pretrain:
    checkpoint = torch.load('./logs/msa-3d-sam-btcv_2023_11_08_13_05_29/Model/latest_checkpoint')
    net.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    best_tol = checkpoint['best_tol']
    starting_epoch = checkpoint['epoch']
    checkpoint = None
else:
    best_tol = 1e4

args.path_helper = set_log_dir('logs', args.exp_name)
logger = create_logger(args.path_helper['log_path'])
logger.info(args)

train_dataloader, valid_dataloader = get_train_dataset(args)

'''checkpoint path and tensorboard'''
# iter_per_epoch = len(Glaucoma_training_loader)
checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)
#use tensorboard
if not os.path.exists(settings.LOG_DIR):
    os.mkdir(settings.LOG_DIR)
writer = SummaryWriter(log_dir=os.path.join(
        settings.LOG_DIR, args.net, settings.TIME_NOW))
# input_tensor = torch.Tensor(args.b, 3, 256, 256).cuda(device = GPUdevice)
# writer.add_graph(net, Variable(input_tensor, requires_grad=True))

#create checkpoint folder to save model
if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)
checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

'''begain training'''
organ_list = ['Liver', 'Lung Cancer', 'Pancreas', 'Hepatic Vessels', 'Spleen', 'Colon Cancer']

for epoch in range(starting_epoch, settings.EPOCH):
    if args.mod == 'sam_adpt':
        net.train()
        time_start = time.time()
        loss, dice, organ_pos_dice_list, organ_neg_dice_list, organ_dice_list = function.train_sam(args, net, optimizer, train_dataloader, epoch, writer, vis = args.vis)
        logger.info(f'----------------------- Train Epoch: {epoch} -----------------------')
        logger.info(f'Train loss: {loss}, dice: {dice}')
        for organ in organ_list:
            logger.info(f'-------------------------{organ}------------------------')
            logger.info(f'organ_dice : {sum(organ_dice_list[organ]) / len(organ_dice_list[organ])}.')
            logger.info(f'pos_dice : {sum(organ_pos_dice_list[organ]) / len(organ_pos_dice_list[organ])}.')
            logger.info(f'neg_dice : {sum(organ_neg_dice_list[organ]) / len(organ_neg_dice_list[organ])}.')
        time_end = time.time()
        print('time_for_training ', time_end - time_start)
        
        net.eval()
        if epoch and epoch % args.val_freq == 0 or epoch == settings.EPOCH-1:
            tol, val_dice, organ_pos_dice_list, organ_neg_dice_list, organ_dice_list = function.validation_sam(args, valid_dataloader, epoch, net, writer)
            logger.info(f'----------------------- Valid Epoch: {epoch} -----------------------')
            logger.info(f'Valid loss: {tol}, dice: {val_dice}')
            for organ in organ_list:
                logger.info(f'-------------------------{organ}------------------------')
                logger.info(f'organ_dice : {sum(organ_dice_list[organ]) / len(organ_dice_list[organ])}.')
                logger.info(f'pos_dice : {sum(organ_pos_dice_list[organ]) / len(organ_pos_dice_list[organ])}.')
                logger.info(f'neg_dice : {sum(organ_neg_dice_list[organ]) / len(organ_neg_dice_list[organ])}.')
            # logger.info(f'Total score: {tol}, DICE: {edice}, IOU: {eiou}, RKID_PF: {organ_dice_list[0]} & {organ_iou_list[0]}, LIVER_PF: {organ_dice_list[1]} & {organ_iou_list[1]}, SPLEEN_PF: {organ_dice_list[2]} & {organ_iou_list[2]}, PANCREAS_PF: {organ_dice_list[3]} & {organ_iou_list[3]} || @ epoch {epoch}.')
            if args.distributed != 'none':
                sd = net.module.state_dict()
            else:
                sd = net.state_dict()

            if tol < best_tol:
                best_tol = tol
                is_best = True
                print('saving model')
                save_checkpoint({
                    'epoch': epoch + 1,
                    'model': args.net,
                    'state_dict': sd,
                    'optimizer': optimizer.state_dict(),
                    'best_tol': best_tol,
                    'path_helper': args.path_helper,
                    'scheduler': scheduler.state_dict(),
                }, is_best, args.path_helper['ckpt_path'], filename="best_checkpoint")
            else:
                is_best = False
        sd = net.state_dict()
        torch.save({
                'epoch': epoch + 1,
                'model': args.net,
                'state_dict': sd,
                'optimizer': optimizer.state_dict(),
                'best_tol': best_tol,
                'path_helper': args.path_helper,
                'scheduler': scheduler.state_dict(),
            }, os.path.join(args.path_helper['ckpt_path'], "latest_checkpoint"))
        # torch.cuda.empty_cache()
        # if epoch and epoch % args.eval_freq == 0 or epoch == settings.EPOCH-1:
        #     logger.info('-------------------------EVALUATION------------------------')
        #     overall_dice, pos_pf, neg_pf = function.eval_sam(args, eval_train_dataloader, epoch, net, writer)
        #     logger.info(f'Train || overall_dice : {overall_dice} pos_pf : {pos_pf} neg_pf : {neg_pf} || @ epoch {epoch}.')
        #     val_overall_dice, val_pos_pf, val_neg_pf = function.eval_sam(args, eval_valid_dataloader, epoch, net, writer)
        #     logger.info(f'Valid || overall_dice : {val_overall_dice} pos_pf : {val_pos_pf} neg_pf : {val_neg_pf} || @ epoch {epoch}.')
        #     (eiou, edice) = function.org_sam(args, eval_train_dataloader, epoch, net, writer)
        #     logger.info(f'Train_org || IOU: {eiou}, DICE: {edice} || @ epoch {epoch}.')
        #     (val_eiou, val_edice) = function.org_sam(args, eval_valid_dataloader, epoch, net, writer)
        #     logger.info(f'Valid_org || IOU: {val_eiou}, DICE: {val_edice} || @ epoch {epoch}.')

writer.close()



# # depth_list = []
# # for batch_idx, pack in enumerate(train_eval_loader):
# #     imgs = pack['image']
# #     depth_list.append(imgs.shape[4])
# # for batch_idx, pack in enumerate(val_eval_loader):
# #     imgs = pack['image']
# #     depth_list.append(imgs.shape[4])
# # import matplotlib.pyplot as plt

# # count_less_than_64 = sum(1 for elem in depth_list if elem < 64)
# # print('count', count_less_than_64, len(depth_list))



# # # 计算每个元素的数量
# # unique_elements = list(set(depth_list))
# # counts = [depth_list.count(elem) for elem in unique_elements]

# # # 创建折线图
# # plt.bar(unique_elements, counts)

# # # 设置 x 轴和 y 轴标签
# # plt.xlabel('Element Values')
# # plt.ylabel('Count')

# # # 显示图形
# # plt.show()
