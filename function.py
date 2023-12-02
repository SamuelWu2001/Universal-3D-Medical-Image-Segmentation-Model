
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
import math
from tensorboardX import SummaryWriter
#from models.discriminatorlayer import discriminator
from conf import settings
import time
import cfg
from conf import settings
from tqdm import tqdm
from utils import *
import torch.nn.functional as F
import torch
from einops import rearrange
import pytorch_ssim
import models.sam.utils.transforms as samtrans
from sklearn.metrics import f1_score

# from lucent.modelzoo.util import get_model_layers
# from lucent.optvis import render, param, transform, objectives
# from lucent.modelzoo import inceptionv1

import shutil
import tempfile

import matplotlib.pyplot as plt
from tqdm import tqdm

from monai.losses import DiceCELoss, DiceFocalLoss
from monai.inferers import sliding_window_inference
from monai.transforms import (
    AsDiscrete,
)
from monai.metrics import DiceMetric


import torch
from patch import patch
import statistics
import clip

args = cfg.parse_args()

GPUdevice = torch.device('cuda', args.gpu_device)
pos_weight = torch.ones([1]).cuda(device=GPUdevice)*2
criterion_G = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
seed = torch.randint(1,11,(args.b,7))

torch.backends.cudnn.benchmark = True
loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
bce_lossfunc = nn.BCELoss()
scaler = torch.cuda.amp.GradScaler()
max_iterations = settings.EPOCH
post_label = AsDiscrete(to_onehot=14)
post_pred = AsDiscrete(argmax=True, to_onehot=14)
dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
dice_val_best = 0.0
global_step_best = 0
epoch_loss_values = []
metric_values = []

clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

def train_sam(args, net: nn.Module, optimizer, train_loader,
          epoch, writer, schedulers=None, vis = 50):
    hard = 0
    epoch_loss = 0
    ind = 0
    # train mode
    net.train()
    optimizer.zero_grad()

    GPUdevice = torch.device('cuda:' + str(args.gpu_device))
    device = GPUdevice
    accumulation_steps = 8/args.b  # 梯度累积的步数
    accumulated_gradients = 0 # 計算當前累積步數
    intersection_total = 0
    union_total = 0
    seg_dice_total = 0
    seg_iou_total = 0
    seg_dice_list = []
    seg_iou_list = []
    pos_print = True
    neg_print = True
    first_pos_index = -1
    first_neg_index = -1
    training_dice = 0
    organ_print_record = {
        'Liver' : 2,
        'Lung Cancer' : 2,
        'Pancreas' : 2,
        'Hepatic Vessels' : 2,
        'Spleen' : 2,
        'Colon Cancer' : 2,
    }
    organ_dice_list = {
        'Liver' : [],
        'Lung Cancer' : [],
        'Pancreas' : [],
        'Hepatic Vessels' : [],
        'Spleen' : [],
        'Colon Cancer' : [],
    }
    organ_pos_dice_list = {
        'Liver' : [],
        'Lung Cancer' : [],
        'Pancreas' : [],
        'Hepatic Vessels' : [],
        'Spleen' : [],
        'Colon Cancer' : [],
    }
    organ_neg_dice_list = {
        'Liver' : [],
        'Lung Cancer' : [],
        'Pancreas' : [],
        'Hepatic Vessels' : [],
        'Spleen' : [],
        'Colon Cancer' : [],
    }

    if args.thd:
        # lossfunc = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
        lossfunc = DiceFocalLoss(sigmoid=True, squared_pred=True, reduction='mean') #, lambda_dice = 1.0, lambda_focal = 10.0, gamma=3.0, alpha=0.8
    else:
        lossfunc = criterion_G

    with tqdm(total=len(train_loader), desc=f'Epoch {epoch}', unit='img') as pbar:
        for batch_idx, pack in enumerate(train_loader):
            imgs = pack['image'].to(dtype = torch.float32, device = GPUdevice)
            masks = pack['label'].to(dtype = torch.float32, device = GPUdevice)
            texts = pack['text']
            output_shape = (256, 256, 64)
            pos_dice_list = []
            neg_dice_list = []
            overall_dice_list = []

            imgs = F.interpolate(imgs, size=output_shape, mode='trilinear', align_corners=False)
            masks = F.interpolate(masks, size=output_shape, mode='trilinear', align_corners=False)
            # print('check', imgs.shape, output_tensor.shape, texts)

            imgs, pt, masks, pos_neg, bboxes = generate_click_prompt(imgs, masks)  
            bboxes = torch.as_tensor(bboxes, dtype=torch.float, device=GPUdevice)

            if args.thd:
                pt = rearrange(pt, 'b n d -> (b d) n')
                pos_neg = rearrange(pos_neg, 'b d -> (b d)')
                imgs = rearrange(imgs, 'b c h w d -> (b d) c h w ')
                masks = rearrange(masks, 'b c h w d -> (b d) c h w ')
                imgs = imgs.repeat(1,3,1,1)
                point_labels = torch.ones(imgs.size(0))
                # imgs = torchvision.transforms.Resize((args.image_size,args.image_size))(imgs)
                # masks = torchvision.transforms.Resize((args.out_size,args.out_size))(masks)

            # showp = pt
            # if torch.sum(pos_neg) != 0 and torch.sum(pos_neg) != args.chunk :
            #     first_pos_index = torch.nonzero(pos_neg == 1)[0].item()
            #     first_neg_index = torch.nonzero(pos_neg == 0)[0].item()


            mask_type = torch.float32
            ind += 1
            b_size,c,w,h = imgs.size()
            longsize = w if w >=h else h

            if point_labels[0] != -1:
                # point_coords = samtrans.ResizeLongestSide(longsize).apply_coords(pt, (h, w))
                point_coords = pt
                coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=GPUdevice)
                labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=GPUdevice)
                coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
                pt = (coords_torch, labels_torch)

            '''init'''
            if hard:
                true_mask_ave = (true_mask_ave > 0.5).float()
                #true_mask_ave = cons_tensor(true_mask_ave)
            imgs = imgs.to(dtype = mask_type,device = GPUdevice)
            
            '''Train'''
            # for n, value in net.image_encoder.named_parameters():
            #     if "Adapter" not in n:
            #         value.requires_grad = False
            imge= net.image_encoder(imgs)

            with torch.no_grad():
                texts_aug = []
                texts_aug.append('A computerized tomography of a '+texts[0])
                text_tolken = clip.tokenize(texts_aug).to(device)
                texte = clip_model.encode_text(text_tolken)
                texte = texte.unsqueeze(1)
                texte = texte.to(torch.float32)
                se, de = net.prompt_encoder(
                    text=None,
                    points=pt,
                    boxes=None,
                    masks=None,
                )
            pred, _ = net.mask_decoder(
                image_embeddings=imge,
                image_pe=net.prompt_encoder.get_dense_pe(), 
                sparse_prompt_embeddings=se,
                dense_prompt_embeddings=de, 
                multimask_output=False,
              )
            # post process
            pred = F.interpolate(
                pred,
                (masks.shape[2], masks.shape[3]),
                mode="bilinear",
                align_corners=False,
            ) 
            loss, dice_loss, ce_loss= lossfunc(pred, masks)

            sigmoid_pred = torch.sigmoid(pred)
            binary_pred = (sigmoid_pred > 0.5).float()
            binary_masks = (masks > 0.5).float()
            dice_score_list = dice_coeff(binary_pred[:,0,:,:], binary_masks[:,0,:,:]) 
            for i in range(len(pos_neg)):
                if pos_neg[i] == 1:
                    pos_dice_list.append(dice_score_list[i])
                elif pos_neg[i] == 0:
                    neg_dice_list.append(dice_score_list[i])
                overall_dice_list.append(dice_score_list[i])   
            training_dice += sum(dice_score_list) / len(dice_score_list)
            organ_dice_list[texts[0]].extend(overall_dice_list)
            organ_pos_dice_list[texts[0]].extend(pos_dice_list)
            organ_neg_dice_list[texts[0]].extend(neg_dice_list)

            if epoch>=20 and epoch % 5 == 0 and organ_print_record[texts[0]] > 0: 
                draw_cal_volume(pred[:,0,:,:].permute(1, 2, 0), masks[:,0,:,:].permute(1, 2, 0), imgs[:,0,:,:].permute(1, 2, 0), batch_idx, epoch, 'train', texts[0], dice_score_list, bboxes)
                organ_print_record[texts[0]] -= 1

            pbar.set_postfix(**{'loss (batch)': loss.item(), 'dice_loss': dice_loss.item(), 'focal_loss': ce_loss.item()}) #, 'focal_loss': ce_loss.item()
            epoch_loss += loss.item()
            loss.backward()

            # nn.utils.clip_grad_value_(net.parameters(), 0.1)
            # optimizer.step()
            # optimizer.zero_grad()
            accumulated_gradients += 1
            if accumulated_gradients == accumulation_steps:
            # 当累积的步数达到设定的步数时，应用累积的梯度更新参数
                optimizer.step()
                optimizer.zero_grad()
                accumulated_gradients = 0  # 重置累积的梯度
                # epoch_loss += loss.item()

            '''vis images'''
            # if vis:
            #     if ind % vis == 0:
            #         namecat = 'Train'
            #         for na in name:
            #             namecat = namecat + na.split('/')[-1].split('.')[0] + '+'
            #         vis_image(imgs,pred,masks, os.path.join(args.path_helper['sample_path'], namecat+'epoch+' +str(epoch) + '.jpg'), reverse=False, points=showp)

            pbar.update()
            # torch.cuda.empty_cache()
    # print('dice_score', (2.0 * intersection_total) / (union_total + 1e-10), intersection_total, union_total)
    # dice_score_avg = (2.0 * intersection_total) / (union_total + 1e-10)
    # iou_score_avg = intersection_total / (union_total - intersection_total)
    # seg_dice_avg = seg_dice_total / len(train_loader)
    # seg_iou_avg = seg_iou_total / len(train_loader) 
    # # print('epoch_loss', epoch_loss, epoch_loss/len(train_loader))
    # seg_dice_median = statistics.median(seg_dice_list)
    # seg_iou_median = statistics.median(seg_iou_list)
    # seg_dice_std = statistics.stdev(seg_dice_list)
    # seg_iou_std = statistics.stdev(seg_iou_list)
    return epoch_loss/len(train_loader), training_dice/len(train_loader), organ_pos_dice_list, organ_neg_dice_list, organ_dice_list #, dice_score_avg, seg_dice_avg, seg_dice_median, seg_dice_std, iou_score_avg, seg_iou_avg, seg_iou_median, seg_iou_std

def test_sam(args, val_loader, epoch, net: nn.Module, clean_dir=True):
     # eval mode
    net.eval()

    mask_type = torch.float32
    n_val = len(val_loader)  # the number of batch
    ave_res, mix_res = (0,0,0,0), (0,0,0,0)
    rater_res = [(0,0,0,0) for _ in range(6)]
    tot = 0
    hard = 0
    threshold = (0.1, 0.3, 0.5, 0.7, 0.9)
    GPUdevice = torch.device('cuda:' + str(args.gpu_device))
    device = GPUdevice
    dice_score = 0
    intersection_total = 0
    union_total = 0

    with tqdm(total=n_val, desc='test round', unit='batch', leave=False) as pbar:
        for ind, pack in enumerate(val_loader):
            imgsw = pack['image'].to(dtype = torch.float32, device = GPUdevice)
            masksw = pack['label'].to(dtype = torch.float32, device = GPUdevice)
            # for k,v in pack['image_meta_dict'].items():
            #     print(k)
            # name = pack['image_meta_dict']['filename_or_obj']
            # 切 patch
            imgsw = imgsw.squeeze()
            masksw = masksw.squeeze()
            imgsw_np = imgsw.cpu().numpy()
            masksw_np = masksw.cpu().numpy()
            patch_size = (64, 64, 64)
            img_patch = patch(imgsw_np, patch_size, -175)
            mask_patch = patch(masksw_np, patch_size, 0)
            imgsw = torch.tensor(img_patch)
            masksw = torch.tensor(mask_patch)


            # print('point', pack['pt'])
            

            for i in range(imgsw.shape[0]):
                imgs = imgsw[i]
                masks = masksw[i]
                imgs, pt, masks, pos_neg = generate_click_prompt(imgs, masks, args.plabel)
                if args.thd:
                    pt = rearrange(pt, 'b n d -> (b d) n')
                    pos_neg = rearrange(pos_neg, 'b d -> (b d)')
                    imgs = rearrange(imgs, 'b c h w d -> (b d) c h w ')
                    masks = rearrange(masks, 'b c h w d -> (b d) c h w ')
                    imgs = imgs.repeat(1,3,1,1)
                    point_labels = torch.ones(imgs.size(0))
                    imgs = torchvision.transforms.Resize((args.image_size,args.image_size))(imgs)
                    masks = torchvision.transforms.Resize((args.out_size,args.out_size))(masks)
                showp = pt

                mask_type = torch.float32
                # ind += 1
                b_size,c,w,h = imgs.size()
                longsize = w if w >=h else h

                if point_labels[0] != -1:
                    # point_coords = samtrans.ResizeLongestSide(longsize).apply_coords(pt, (h, w))
                    point_coords = pt
                    coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=GPUdevice)
                    labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=GPUdevice)
                    coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
                    pt = (coords_torch, labels_torch)

                '''init'''
                if hard:
                    true_mask_ave = (true_mask_ave > 0.5).float()
                    #true_mask_ave = cons_tensor(true_mask_ave)
                imgs = imgs.to(dtype = mask_type,device = GPUdevice)
                masks = masks.to(dtype = mask_type,device = GPUdevice)
                
                '''test'''
                with torch.no_grad():
                    imge= net.image_encoder(imgs)
                    se, de = net.prompt_encoder(
                        points=pt,
                        boxes=None,
                        masks=None,
                    )
                    pred, _ = net.mask_decoder(
                        image_embeddings=imge,
                        image_pe=net.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=se,
                        dense_prompt_embeddings=de, 
                        multimask_output=False,
                    )
                    intersection, union, _ = cal_intersection_union(pred, masks)
                    intersection_total += intersection
                    union_total += union
                    print('dice_score', (2.0 * intersection) / (union + 1e-10), intersection, union)
                    '''vis images'''
                    # if ind % args.vis == 0:
                    #     namecat = 'Test'
                    #     for na in name:
                    #         img_name = na.split('/')[-1].split('.')[0]
                    #         namecat = namecat + img_name + '+'
                    #     vis_image(imgs,pred, masks, os.path.join(args.path_helper['sample_path'], namecat+'epoch+' +str(epoch) + '.jpg'), reverse=False, points=showp)
                    

                    temp = eval_seg(pred, masks, threshold)
                    mix_res = tuple([sum(a) for a in zip(mix_res, temp)])
            pbar.update()
    dice_score =  (2.0 * intersection_total) / (union_total + 1e-10)   
    # if args.evl_chunk:
    #     n_val = n_val * (imgsw.size(-1) // evl_ch)

    return dice_score

def validation_sam(args, val_loader, epoch, net: nn.Module, clean_dir=True):
     # eval mode
    net.eval()

    mask_type = torch.float32
    n_val = len(val_loader)  # the number of batch
    ave_res, mix_res = (0,0,0,0), (0,0,0,0)
    rater_res = [(0,0,0,0) for _ in range(6)]
    tot = 0
    hard = 0
    threshold = (0.1, 0.3, 0.5, 0.7, 0.9)
    GPUdevice = torch.device('cuda:' + str(args.gpu_device))
    device = GPUdevice
    epoch_loss = 0
    pos_print = True
    neg_print = True
    first_pos_index = -1
    first_neg_index = -1
    valid_dice = 0
    organ_print_record = {
        'Liver' : 4,
        'Lung Cancer' : 4,
        'Pancreas' : 4,
        'Hepatic Vessels' : 4,
        'Spleen' : 4,
        'Colon Cancer' : 4,
    }
    organ_dice_list = {
        'Liver' : [],
        'Lung Cancer' : [],
        'Pancreas' : [],
        'Hepatic Vessels' : [],
        'Spleen' : [],
        'Colon Cancer' : [],
    }
    organ_pos_dice_list = {
        'Liver' : [],
        'Lung Cancer' : [],
        'Pancreas' : [],
        'Hepatic Vessels' : [],
        'Spleen' : [],
        'Colon Cancer' : [],
    }
    organ_neg_dice_list = {
        'Liver' : [],
        'Lung Cancer' : [],
        'Pancreas' : [],
        'Hepatic Vessels' : [],
        'Spleen' : [],
        'Colon Cancer' : [],
    }

    if args.thd:
        # lossfunc = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
        lossfunc = DiceFocalLoss(sigmoid=True, squared_pred=True, reduction='mean') #, lambda_dice = 1.0, lambda_focal = 10.0, gamma=3.0, alpha=0.8
    else:
        lossfunc = criterion_G

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch_idx, pack in enumerate(val_loader):
            imgs = pack['image'].to(dtype = torch.float32, device = GPUdevice)
            masks = pack['label'].to(dtype = torch.float32, device = GPUdevice)
            texts = pack['text']
            output_shape = (256, 256, 64)
            imgs = F.interpolate(imgs, size=output_shape, mode='trilinear', align_corners=False)
            masks = F.interpolate(masks, size=output_shape, mode='trilinear', align_corners=False)
            pos_dice_list = []
            neg_dice_list = []
            overall_dice_list = []
            
            imgs, pt, masks, pos_neg, bboxes = generate_click_prompt(imgs, masks)  
            bboxes = torch.as_tensor(bboxes, dtype=torch.float, device=GPUdevice)

            if args.thd:
                pt = rearrange(pt, 'b n d -> (b d) n')
                pos_neg = rearrange(pos_neg, 'b d -> (b d)')
                imgs = rearrange(imgs, 'b c h w d -> (b d) c h w ')
                masks = rearrange(masks, 'b c h w d -> (b d) c h w ')
                imgs = imgs.repeat(1,3,1,1)
                point_labels = torch.ones(imgs.size(0))
                
            showp = pt

            if torch.sum(pos_neg) != 0 and torch.sum(pos_neg) != args.chunk :
                first_pos_index = torch.nonzero(pos_neg == 1)[0].item()
                first_neg_index = torch.nonzero(pos_neg == 0)[0].item()

            mask_type = torch.float32
            b_size,c,w,h = imgs.size()
            longsize = w if w >=h else h

            if point_labels[0] != -1:
                # point_coords = samtrans.ResizeLongestSide(longsize).apply_coords(pt, (h, w))
                point_coords = pt
                coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=GPUdevice)
                labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=GPUdevice)
                coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
                pt = (coords_torch, labels_torch)

            '''init'''
            if hard:
                true_mask_ave = (true_mask_ave > 0.5).float()
                #true_mask_ave = cons_tensor(true_mask_ave)
            imgs = imgs.to(dtype = mask_type,device = GPUdevice)
            masks = masks.to(dtype = mask_type,device = GPUdevice)
            
            '''Validate'''
            # for n, value in net.image_encoder.named_parameters():
            #     if "Adapter" not in n:
            #         value.requires_grad = False
            # imge= net.image_encoder(imgs)

            with torch.no_grad():
                imge= net.image_encoder(imgs)
                texts_aug = []
                texts_aug.append('A computerized tomography of a '+texts[0])
                text_tolken = clip.tokenize(texts_aug).to(device)
                texte = clip_model.encode_text(text_tolken)
                texte = texte.unsqueeze(1)
                texte = texte.to(torch.float32)
                se, de = net.prompt_encoder(
                    text=None,
                    points=pt,
                    boxes=None,
                    masks=None,
                )
                pred, _ = net.mask_decoder(
                    image_embeddings=imge,
                    image_pe=net.prompt_encoder.get_dense_pe(), 
                    sparse_prompt_embeddings=se,
                    dense_prompt_embeddings=de, 
                    multimask_output=False,
                )
                # print('seg', pred.shape, masks.shape)
                # post process
                pred = F.interpolate(
                    pred,
                    (masks.shape[2], masks.shape[3]),
                    mode="bilinear",
                    align_corners=False,
                ) 
            
            loss, dice_loss, ce_loss= lossfunc(pred, masks)
            # bce_loss = bce_lossfunc(torch.sigmoid(pred), masks)
            # loss = dice_loss + bce_loss
            sigmoid_pred = torch.sigmoid(pred)
            binary_pred = (sigmoid_pred > 0.5).float()
            binary_masks = (masks > 0.5).float()
            dice_score_list = dice_coeff(binary_pred[:,0,:,:], binary_masks[:,0,:,:])
            valid_dice += sum(dice_score_list) / len(dice_score_list)
            for i in range(len(pos_neg)):
                if pos_neg[i] == 1:
                    pos_dice_list.append(dice_score_list[i])
                elif pos_neg[i] == 0:
                    neg_dice_list.append(dice_score_list[i])
                overall_dice_list.append(dice_score_list[i])
            organ_dice_list[texts[0]].extend(overall_dice_list)
            organ_pos_dice_list[texts[0]].extend(pos_dice_list)
            organ_neg_dice_list[texts[0]].extend(neg_dice_list)

            if epoch>=20 and organ_print_record[texts[0]] > 0:
                draw_cal_volume(pred[:,0,:,:].permute(1, 2, 0), masks[:,0,:,:].permute(1, 2, 0), imgs[:,0,:,:].permute(1, 2, 0), batch_idx, epoch, 'valid', texts[0], dice_score_list, bboxes)
                organ_print_record[texts[0]] -= 1
            
            pbar.set_postfix(**{'loss (batch)': loss.item(), 'dice_loss': dice_loss.item(), 'focal_loss': ce_loss.item()}) #, 'focal_loss': ce_loss.item()
            epoch_loss += loss.item()
            
            pbar.update()
            # torch.cuda.empty_cache()
    # plabel_dice_avg = [dice / n_val for dice in plabel_dice_total]
    # plabel_iou_avg = [iou / n_val for iou in plabel_iou_total]
    return epoch_loss / len(val_loader), valid_dice / len(val_loader), organ_pos_dice_list, organ_neg_dice_list, organ_dice_list #, iou_score_avg, dice_score_avg #plabel_dice_avg, plabel_iou_avg


def eval_sam_test(args, eval_loader, epoch, net: nn.Module, clean_dir=True):
    # eval mode
    net.eval()

    mask_type = torch.float32
    n_val = len(eval_loader)  # the number of batch
    hard = 0
    threshold = (0.1, 0.3, 0.5, 0.7, 0.9)
    GPUdevice = torch.device('cuda:' + str(args.gpu_device))
    device = GPUdevice
    plabel = 1
    patch_num_list = []
    pos_num_list = []
    neg_num_list = []
    negative_correct_num_list = []
    positive_correct_num_list = []
    pos_dice_list = []
    # pos_pred_area_list = []
    # neg_pred_area_list = []
    # decathlon
    ct_min_norm = {
        'Liver' : -2.96,
        'Lung Cancer' : -2.67,
        'Pancreas' : -2.31,
        'Hepatic Vessels' : -2.04,
        'Spleen' : -3.55,
        'Colon Cancer' : -2.82,
    }
    # BTCV
    # ct_min_norm = {
    #     "spleen" : -4.43,
    #     "kidney right" : -4.15,
    #     "kidney left" : -3.85,
    #     "gallbladder" : -2.11,
    #     "esophagus" : -5.16,
    #     "liver" : -4.59,
    #     "stomach" : -2.79,
    #     "aorta" : -4.14,
    #     "inferior vena cava" : -3.95,
    #     "portal vein and splenic vein" : -3.87,
    #     "pancreas" : -3.35,
    #     "adrenal gland" : -2.74,
    # }

    with tqdm(total=n_val, desc='evaluate round', unit='batch', leave=False) as pbar:
        total_dice_patch_list = []
        

        for ind, pack in enumerate(eval_loader):
            imgsw = pack['image'].to(dtype = torch.float32, device = GPUdevice)
            masksw = pack['label'].to(dtype = torch.float32, device = GPUdevice)
            texts = pack['text']

            total_intersection_total = 0
            total_union_total = 0

            imgsw = imgsw.squeeze()
            masksw = masksw.squeeze()
            imgsw_np = imgsw.cpu().numpy()
            org_masksw = masksw
            print('imgsw_np', imgsw_np.shape, masksw.shape)

            x_patch_num = math.ceil(imgsw_np.shape[0] / args.roi_size)
            y_patch_num = math.ceil(imgsw_np.shape[1] / args.roi_size)
            z_patch_num = math.ceil(imgsw_np.shape[2] / args.evl_chunk)

            masksw_np = masksw.cpu().numpy()
            patch_size = (args.roi_size, args.roi_size, args.evl_chunk)
            img_patch = patch(imgsw_np, patch_size, ct_min_norm[texts[0]])
            mask_patch = patch(masksw_np, patch_size, 0)
            imgsw = torch.tensor(img_patch)
            masksw = torch.tensor(mask_patch)
            patch_num_list.append(imgsw.shape[0])

            pos_num = 0
            pos_intersection_total = 0
            pos_union_total = 0
            pos_pred_area = 0
            neg_intersection_total = 0
            neg_union_total = 0
            neg_num = 0
            negative_correct_num = 0
            positive_correct_num = 0
            neg_pred_area = 0

            intersection_per_vol_list = []
            union_per_vol_list = []
            total_dice_patch = []
            pred_patch_list = []

            for i in range(imgsw.shape[0]):
                imgs = imgsw[i]
                masks = masksw[i]
                imgs, pt, masks, pos_neg = generate_click_prompt(imgs, masks, plabel)
                if args.thd:
                    pt = rearrange(pt, 'b n d -> (b d) n')
                    pos_neg = rearrange(pos_neg, 'b d -> (b d)')
                    imgs = rearrange(imgs, 'b c h w d -> (b d) c h w ')
                    masks = rearrange(masks, 'b c h w d -> (b d) c h w ')
                    imgs = imgs.repeat(1,3,1,1)
                    point_labels = torch.ones(imgs.size(0))

                mask_type = torch.float32

                if point_labels[0] != -1:
                    # point_coords = samtrans.ResizeLongestSide(longsize).apply_coords(pt, (h, w))
                    point_coords = pt
                    coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=GPUdevice)
                    labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=GPUdevice)
                    coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
                    pt = (coords_torch, labels_torch)
                    
                imgs = imgs.to(dtype = mask_type,device = GPUdevice)
                masks = masks.to(dtype = mask_type,device = GPUdevice)

                '''test'''
                with torch.no_grad():
                    imge= net.image_encoder(imgs)
                    se, de = net.prompt_encoder(
                        points=pt,
                        boxes=None,
                        masks=None,
                    )
                    text_tolken = clip.tokenize(texts).to(device)
                    texte = clip_model.encode_text(text_tolken)
                    texte = texte.unsqueeze(1)
                    texte = texte.to(torch.float32)
                    texte = text_prompt_embedding(texte)
                    pred, _ = net.mask_decoder(
                        image_embeddings=imge,
                        image_pe=net.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=se,
                        dense_prompt_embeddings=de, 
                        multimask_output=False,
                    )
                    # post process
                    pred = F.interpolate(
                        pred,
                        (args.out_size, args.out_size),
                        mode="bilinear",
                        align_corners=False,
                    ) 
                    pred_patch_list.append(pred.squeeze().permute(1, 2, 0))
                    # comaper slice based 3d dice 
                    # metric = DiceMetric(include_background=True, reduction="mean")
                    # d1 = cal_3dslice_dice(pred, masks)
                    # binary_y_true = (masks > 0.5).float()
                    # sigmoid_pred = torch.sigmoid(pred)
                    # binary_y_pred = (sigmoid_pred >= 0.5).float()
                    # d2 = metric(binary_y_pred, binary_y_true)
                    # print('compare', d1, d2)
                    # print('pos_neg', pos_neg.shape, pos_neg)
                    # masks = (masks > 0.5).float()
                    # sigmoid_pred = torch.sigmoid(pred)
                    # binary_y_pred = (sigmoid_pred > 0.5).float()
                    # dice_score_list = dice_coeff(binary_y_pred[:,0,:,:], masks[:,0,:,:])
                    # dice_score_per_vol_list.extend(dice_score_list)
                    for i in  range(pos_neg.shape[0]) :
                        intersection, union, pred_area = cal_intersection_union(pred[i], masks[i])
                        intersection_per_vol_list.append(intersection)
                        union_per_vol_list.append(union)
                        total_intersection_total += intersection
                        total_union_total += union
                        if  pos_neg[i] == 0:
                            neg_intersection_total += intersection
                            neg_union_total += union
                            neg_pred_area += pred_area
                            neg_num += 1
                            if pred_area == 0:
                                negative_correct_num += 1
                        else :
                            pos_intersection_total += intersection
                            pos_union_total += union
                            pos_pred_area += pred_area
                            pos_num += 1
                            if pred_area > 0:
                                positive_correct_num += 1
            
            pos_dice = (2.0 * pos_intersection_total) / (pos_union_total + 1e-10)
            pos_pred_area = (pos_pred_area / pos_num) / (args.roi_size * args.roi_size)
            neg_pred_area = (neg_pred_area / neg_num) / (args.roi_size * args.roi_size)
            pos_num_list.append(pos_num)
            neg_num_list.append(neg_num)
            negative_correct_num_list.append(negative_correct_num)
            positive_correct_num_list.append(positive_correct_num)
            pos_dice_list.append(pos_dice)
            # pos_pred_area_list.append(pos_pred_area)
            # neg_pred_area_list.append(neg_pred_area)

            print(len(pred_patch_list), pred_patch_list[0].shape)
            for i in range(280, 336):
                pred_mask = pred_patch_list[i]
                sigmoid_pred = torch.sigmoid(pred_mask)
                binary_pred = (sigmoid_pred > 0.5).float()    
                pred_mask = binary_pred.detach().cpu().numpy()[:,:,27]
                # print('pred_mask', pred_mask.shape)
                # 使用 Matplotlib 绘制图像
                plt.figure(figsize=(10, 3))  # 调整图像大小           
                # 绘制第一个张量的图像
                plt.subplot(111)  
                plt.imshow(pred_mask, cmap='gray')
                plt.title('pred_mask')                                
                # 调整子图布局
                plt.tight_layout()            
                # 保存图像
                plt.savefig(f'image/test/{i}.png')
                plt.close()
            gt_masks = org_masksw.detach().cpu().numpy()
            gt_mask = gt_masks[:,:,347]
            # 使用 Matplotlib 绘制图像
            plt.figure(figsize=(10, 3))  # 调整图像大小           
            # 绘制第一个张量的图像
            plt.subplot(121)  # 分割为 1x3 子图，选择第一个子图
            plt.imshow(gt_mask, cmap='gray')
            plt.title('pred_mask') 
            plt.savefig('image/test/mask.png')
            plt.close()
            combined_pred = torch.cat(pred_patch_list, dim=2)
            patch_pred_result = combined_pred.view(512, 448, 640)
            patch_pred_result = patch_pred_result[:org_masksw.shape[0], :org_masksw.shape[1], :org_masksw.shape[2]]
            # draw_cal_volume(patch_pred_result, org_masksw, ind+1)
            
            for k in range(z_patch_num):
                intersection_patch_list = [0] * args.evl_chunk
                union_patch_list = [0] * args.evl_chunk
                for i in range(x_patch_num * y_patch_num):
                    for j in range(args.evl_chunk):
                        intersection_patch_list[j] += intersection_per_vol_list[k*x_patch_num*y_patch_num*args.evl_chunk + i*args.evl_chunk + j]
                        union_patch_list[j] += union_per_vol_list[k*x_patch_num*y_patch_num*args.evl_chunk + i*args.evl_chunk + j]
                        inter_list = [2*x + 0.000001 for x in intersection_patch_list]
                union_list = [x + 0.000001 for x in union_patch_list]
                dice_list = [a / b for a, b in zip(inter_list, union_list)]
                # print('inter_list', inter_list[0:2], intersection_patch_list[0:2])
                # print('union_list', union_list[0:2], union_patch_list[0:2])
                # print('dice_list', dice_list[0:2])
                total_dice_patch.extend(dice_list)
            # print('check', imgsw_np.shape[2], len(total_dice_patch))
            # print(len(total_dice_patch[:imgsw_np.shape[2]]))
            total_dice_patch_list.append(sum(total_dice_patch[:imgsw_np.shape[2]]) / len(total_dice_patch[:imgsw_np.shape[2]]))
            pbar.update()

        
        neg_ratio = sum(neg_num_list) / (sum(pos_num_list) +sum(neg_num_list))
        print('neg / total', neg_ratio) 
        # print('total_dice_patch_list', total_dice_patch_list)
        overall_dice = sum(total_dice_patch_list) / len(total_dice_patch_list)
        print('overall_dice', overall_dice) 

        # performance for different organ
        organ_dice_list = []
        organ_pos_dice_list = []
        # organ_pos_pred_area_list = []
        # organ_neg_pred_area_list = []
        organ_negative_correct_ratio_list = []
        organ_positive_correct_ratio_list = []
        for i in range(6):
            organ_dice = sum(total_dice_patch_list[i*7:i*7+7]) / 7
            organ_pos_dice = sum(pos_dice_list[i*7:i*7+7]) / 7
            # organ_pos_pred_area = sum(pos_pred_area_list[i*7:i*7+7]) / 7
            # organ_neg_pred_area = sum(neg_pred_area_list[i*7:i*7+7]) / 7
            organ_negative_correct_ratio = sum(negative_correct_num_list[i*7:i*7+7]) / sum(neg_num_list[i*7:i*7+7])
            organ_dice_list.append(organ_dice)
            organ_pos_dice_list.append(organ_pos_dice)
            # organ_pos_pred_area_list.append(organ_pos_pred_area)
            # organ_neg_pred_area_list.append(organ_neg_pred_area)
            organ_negative_correct_ratio_list.append(organ_negative_correct_ratio)
        # for i in range(12):
        #     organ_dice = sum(total_dice_patch_list[i*6:i*6+6]) / 6
        #     organ_pos_dice = sum(pos_dice_list[i*6:i*6+6]) / 6
        #     organ_pos_pred_area = sum(pos_pred_area_list[i*6:i*6+6]) / 6
        #     organ_neg_pred_area = sum(neg_pred_area_list[i*6:i*6+6]) / 6
        #     organ_negative_correct_ratio = sum(negative_correct_num_list[i*6:i*6+6]) / sum(neg_num_list[i*6:i*6+6])
        #     organ_positive_correct_ratio = sum(positive_correct_num_list[i*6:i*6+6]) / sum(pos_num_list[i*6:i*6+6])
        #     organ_dice_list.append(organ_dice)
        #     organ_pos_dice_list.append(organ_pos_dice)
        #     organ_pos_pred_area_list.append(organ_pos_pred_area)
        #     organ_neg_pred_area_list.append(organ_neg_pred_area)
        #     organ_negative_correct_ratio_list.append(organ_negative_correct_ratio)
        #     organ_positive_correct_ratio_list.append(organ_positive_correct_ratio)



            
    return organ_dice_list, organ_pos_dice_list, organ_negative_correct_ratio_list, organ_positive_correct_ratio_list

def org_sam(args, val_loader, epoch, net: nn.Module, clean_dir=True):
    net.eval()

    mask_type = torch.float32
    # ave_res, mix_res = (0,0,0,0), (0,0,0,0)
    rater_res = [(0,0,0,0) for _ in range(6)]
    hard = 0
    threshold = (0.1, 0.3, 0.5, 0.7, 0.9)
    GPUdevice = torch.device('cuda:' + str(args.gpu_device))
    device = GPUdevice
    n_val = len(val_loader)
    iou_list = []
    dice_list =[]

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for ind, pack in enumerate(val_loader):
            mix_res = (0,0,0,0)
            patch_num = 0
            imgsw = pack['image'].to(dtype = torch.float32, device = GPUdevice)
            masksw = pack['label'].to(dtype = torch.float32, device = GPUdevice)
            texts = pack['text']
            # for k,v in pack['image_meta_dict'].items():
            #     print(k)
            if 'pt' not in pack:
                imgsw, ptw, masksw, pos_neg = generate_click_prompt(imgsw, masksw)
                pos_neg = rearrange(pos_neg, 'b d -> (b d)')
            else:
                ptw = pack['pt']
                point_labels = pack['p_label']
            
            buoy = 0
            if args.evl_chunk:
                evl_ch = int(args.evl_chunk)
            else:
                evl_ch = int(imgsw.size(-1))

            while (buoy + evl_ch) < imgsw.size(-1) + evl_ch:
                if args.thd:
                    pt = ptw[:,:,buoy: buoy + evl_ch]
                else:
                    pt = ptw

                imgs = imgsw[...,buoy:buoy + evl_ch]
                masks = masksw[...,buoy:buoy + evl_ch]
                buoy += evl_ch

                if args.thd:
                    pt = rearrange(pt, 'b n d -> (b d) n')
                    imgs = rearrange(imgs, 'b c h w d -> (b d) c h w ')
                    masks = rearrange(masks, 'b c h w d -> (b d) c h w ')
                    imgs = imgs.repeat(1,3,1,1)
                    point_labels = torch.ones(imgs.size(0))

                    imgs = torchvision.transforms.Resize((args.image_size,args.image_size))(imgs)
                    masks = torchvision.transforms.Resize((args.out_size,args.out_size))(masks)
                
                mask_type = torch.float32
                ind += 1
                b_size,c,w,h = imgs.size()
                longsize = w if w >=h else h

                if point_labels[0] != -1:
                    # point_coords = samtrans.ResizeLongestSide(longsize).apply_coords(pt, (h, w))
                    point_coords = pt
                    coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=GPUdevice)
                    labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=GPUdevice)
                    coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
                    pt = (coords_torch, labels_torch)

                '''init'''
                if hard:
                    true_mask_ave = (true_mask_ave > 0.5).float()
                    #true_mask_ave = cons_tensor(true_mask_ave)
                imgs = imgs.to(dtype = mask_type,device = GPUdevice)
                
                '''test'''
                with torch.no_grad():
                    imge= net.image_encoder(imgs)

                    se, de = net.prompt_encoder(
                        points=pt,
                        boxes=None,
                        masks=None,
                    )
                    text_tolken = clip.tokenize(texts).to(device)
                    texte = clip_model.encode_text(text_tolken)
                    texte = texte.unsqueeze(1)
                    texte = texte.to(torch.float32)
                    texte = text_prompt_embedding(texte)
                    pred, _ = net.mask_decoder(
                        image_embeddings=imge,
                        image_pe=net.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=texte,
                        dense_prompt_embeddings=de, 
                        multimask_output=False,
                    )
                    # post process
                    pred = F.interpolate(
                        pred,
                        (args.out_size, args.out_size),
                        mode="bilinear",
                        align_corners=False,
                    )

                    temp = eval_seg(pred, masks, threshold)
                    temp = tuple(value * pred.shape[0] for value in temp)
                    patch_num += pred.shape[0]
                    mix_res = tuple([sum(a) for a in zip(mix_res, temp)])

            (iou, dice) = tuple([a/patch_num for a in mix_res])
            iou_list.append(iou)
            dice_list.append(dice)

            pbar.update()
    return ((sum(iou_list) / len(iou_list)), (sum(dice_list) / len(dice_list)))

def org_sam_test(args, val_loader, epoch, net: nn.Module, clean_dir=True):

    net.eval()

    mask_type = torch.float32
    # ave_res, mix_res = (0,0,0,0), (0,0,0,0)
    rater_res = [(0,0,0,0) for _ in range(6)]
    hard = 0
    threshold = (0.1, 0.3, 0.5, 0.7, 0.9)
    GPUdevice = torch.device('cuda:' + str(args.gpu_device))
    device = GPUdevice
    n_val = len(val_loader)
    iou_list = []
    dice_list =[]
    pos_dice_per_vol_list = [] 
    pos_pred_area_per_vol_list = []
    neg_pred_area_per_vol_list = []

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for ind, pack in enumerate(val_loader):
            mix_res = (0,0,0,0)
            patch_num = 0
            imgsw = pack['image'].to(dtype = torch.float32, device = GPUdevice)
            masksw = pack['label'].to(dtype = torch.float32, device = GPUdevice)
            # texts = pack['text']
            texts = ['cat']
            pos_dice_per_patch_list = [] 
            pos_pred_area_per_patch_list = []
            neg_pred_area_per_patch_list = []
            # for k,v in pack['image_meta_dict'].items():
            #     print(k)
            if 'pt' not in pack:
                imgsw, ptw, masksw, pos_neg = generate_click_prompt(imgsw, masksw)
                pos_neg = rearrange(pos_neg, 'b d -> (b d)')
            else:
                ptw = pack['pt']
                point_labels = pack['p_label']
            
            buoy = 0
            if args.evl_chunk:
                evl_ch = int(args.evl_chunk)
            else:
                evl_ch = int(imgsw.size(-1))

            while buoy < imgsw.size(-1):
                if args.thd:
                    pt = ptw[:,:,buoy: buoy + evl_ch]
                else:
                    pt = ptw

                imgs = imgsw[...,buoy:buoy + evl_ch]
                masks = masksw[...,buoy:buoy + evl_ch]

                if args.thd:
                    pt = rearrange(pt, 'b n d -> (b d) n')
                    imgs = rearrange(imgs, 'b c h w d -> (b d) c h w ')
                    masks = rearrange(masks, 'b c h w d -> (b d) c h w ')
                    imgs = imgs.repeat(1,3,1,1)
                    point_labels = torch.ones(imgs.size(0))
                    imgs = torchvision.transforms.Resize((args.image_size,args.image_size))(imgs)
                    masks = torchvision.transforms.Resize((args.out_size,args.out_size))(masks)
                
                mask_type = torch.float32
                ind += 1
                b_size,c,w,h = imgs.size()
                longsize = w if w >=h else h

                if point_labels[0] != -1:
                    # point_coords = samtrans.ResizeLongestSide(longsize).apply_coords(pt, (h, w))
                    point_coords = pt
                    coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=GPUdevice)
                    labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=GPUdevice)
                    coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
                    pt = (coords_torch, labels_torch)

                '''init'''
                if hard:
                    true_mask_ave = (true_mask_ave > 0.5).float()
                    #true_mask_ave = cons_tensor(true_mask_ave)
                imgs = imgs.to(dtype = mask_type,device = GPUdevice)
                
                '''test'''
                with torch.no_grad():
                    imge= net.image_encoder(imgs)

                    se, de = net.prompt_encoder(
                        points=pt,
                        boxes=None,
                        masks=None,
                    )
                    text_tolken = clip.tokenize(texts).to(device)
                    texte = clip_model.encode_text(text_tolken)
                    texte = texte.unsqueeze(1)
                    texte = texte.to(torch.float32)
                    texte = text_prompt_embedding(texte)
                    pred, _ = net.mask_decoder(
                        image_embeddings=imge,
                        image_pe=net.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=texte,
                        dense_prompt_embeddings=de, 
                        multimask_output=False,
                    )
                    # post process
                    pred = F.interpolate(
                        pred,
                        (args.out_size, args.out_size),
                        mode="bilinear",
                        align_corners=False,
                    )
                    temp, pos_dice_list, pos_pred_area_list, neg_pred_area_list = eval_seg(pred, masks, 0.5)
                    pos_dice_per_patch_list.extend(pos_dice_list)
                    pos_pred_area_per_patch_list.extend(pos_pred_area_list)
                    neg_pred_area_per_patch_list.extend(neg_pred_area_list)
                    temp = tuple(value * pred.shape[0] for value in temp)
                    patch_num += pred.shape[0]
                    mix_res = tuple([sum(a) for a in zip(mix_res, temp)])
                
                buoy += evl_ch

            (iou, dice) = tuple([a/patch_num for a in mix_res])
            iou_list.append(iou)
            dice_list.append(dice)
            pos_dice_per_vol_list.append(sum(pos_dice_per_patch_list) / len(pos_dice_per_patch_list))
            pos_pred_area_per_vol_list.append(sum(pos_pred_area_per_patch_list) / len(pos_pred_area_per_patch_list))
            neg_pred_area_per_vol_list.append(sum(neg_pred_area_per_patch_list) / len(neg_pred_area_per_patch_list))

            pbar.update()

    # performance for different organ
    organ_dice_list = []
    organ_iou_list = []
    organ_pos_dice_list = [] 
    organ_pos_pred_area_list = []
    organ_neg_pred_area_list = []
    for i in range(6):
        organ_dice = sum(dice_list[i*7:i*7+7]) / 7
        organ_iou = sum(iou_list[i*7:i*7+7]) / 7
        organ_pos_dice = sum(pos_dice_per_vol_list[i*7:i*7+7]) / 7
        organ_pos_pred_area = sum(pos_pred_area_per_vol_list[i*7:i*7+7]) / 7 / 4096
        organ_neg_pred_area = sum(neg_pred_area_per_vol_list[i*7:i*7+7]) / 7 / 4096
        organ_dice_list.append(organ_dice)
        organ_iou_list.append(organ_iou)
        organ_pos_dice_list.append(organ_pos_dice)
        organ_pos_pred_area_list.append(organ_pos_pred_area)
        organ_neg_pred_area_list.append(organ_neg_pred_area)


    return organ_iou_list, organ_dice_list, organ_pos_dice_list, organ_pos_pred_area_list, organ_neg_pred_area_list

def eval_sam(args, eval_loader, epoch, net: nn.Module, clean_dir=True):
    # eval mode
    net.eval()

    n_val = len(eval_loader)  # the number of batch
    GPUdevice = torch.device('cuda:' + str(args.gpu_device))
    device = GPUdevice
    
    organ_dice_list = {
        'Liver' : [],
        'Lung Cancer' : [],
        'Pancreas' : [],
        'Hepatic Vessels' : [],
        'Spleen' : [],
        'Colon Cancer' : [],
    }
    organ_pos_dice_list = {
        'Liver' : [],
        'Lung Cancer' : [],
        'Pancreas' : [],
        'Hepatic Vessels' : [],
        'Spleen' : [],
        'Colon Cancer' : [],
    }
    organ_neg_dice_list = {
        'Liver' : [],
        'Lung Cancer' : [],
        'Pancreas' : [],
        'Hepatic Vessels' : [],
        'Spleen' : [],
        'Colon Cancer' : [],
    }
    organ_print_record = {
        'Liver' : 2,
        'Lung Cancer' : 2,
        'Pancreas' : 2,
        'Hepatic Vessels' : 2,
        'Spleen' : 2,
        'Colon Cancer' : 2,
    }
    organ_dice = {
        'Liver' : [],
        'Lung Cancer' : [],
        'Pancreas' : [],
        'Hepatic Vessels' : [],
        'Spleen' : [],
        'Colon Cancer' : [],
    }
    with tqdm(total=n_val, desc='evaluate round', unit='batch', leave=False) as pbar:
        for batch_idx, pack in enumerate(eval_loader):
            imgsw = pack['image'].to(dtype = torch.float32, device = GPUdevice)
            masksw = pack['label'].to(dtype = torch.float32, device = GPUdevice)
            texts = pack['text']
            depth = imgsw.shape[4]
            bottom = 0
            pos_dice_list = []
            neg_dice_list = []
            overall_dice_list = []
            concatenated_pred = torch.empty(0, masksw.shape[2], masksw.shape[3]).to(dtype = torch.float32, device = GPUdevice)
            concatenated_mask = torch.empty(0, masksw.shape[2], masksw.shape[3]).to(dtype = torch.float32, device = GPUdevice)
            concatenated_imgs = torch.empty(0, masksw.shape[2], masksw.shape[3]).to(dtype = torch.float32, device = GPUdevice)
            concatenated_point = torch.empty(0, 2).to(dtype = torch.float32, device = GPUdevice)
            while bottom < depth:
                output_shape = (256, 256, min(depth-bottom, 64))
                ori_imgs = imgsw[:,:,:,:,bottom:bottom + min(depth-bottom, 64)]
                imgs = F.interpolate(ori_imgs, size=output_shape, mode='trilinear', align_corners=False)
                ori_masks = masksw[:,:,:,:,bottom:bottom + min(depth-bottom, 64)]
                masks = F.interpolate(ori_masks, size=output_shape, mode='trilinear', align_corners=False)
                if 'pt' not in pack:
                    imgs, pt, masks, pos_neg = generate_click_prompt(imgs, masks)  
                else:
                    pt = pack['pt']
                    point_labels = pack['p_label']

                if args.thd:
                    pt = rearrange(pt, 'b n d -> (b d) n')
                    pos_neg = rearrange(pos_neg, 'b d -> (b d)')
                    imgs = rearrange(imgs, 'b c h w d -> (b d) c h w ')
                    ori_masks = rearrange(ori_masks, 'b c h w d -> (b d) c h w ')
                    imgs = imgs.repeat(1,3,1,1)
                    point_labels = torch.ones(imgs.size(0))
                showp = pt
                concatenated_point = torch.cat((concatenated_point, showp), dim=0)
                if point_labels[0] != -1:
                    # point_coords = samtrans.ResizeLongestSide(longsize).apply_coords(pt, (h, w))
                    point_coords = pt
                    coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=GPUdevice)
                    labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=GPUdevice)
                    coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
                    pt = (coords_torch, labels_torch)
                
                '''test'''
                with torch.no_grad():
                    imge= net.image_encoder(imgs)
                    texts_aug = []
                    texts_aug.append('A computerized tomography of a '+texts[0])
                    text_tolken = clip.tokenize(texts).to(device)
                    texte = clip_model.encode_text(text_tolken)
                    texte = texte.unsqueeze(1)
                    texte = texte.to(torch.float32)
                    se, de = net.prompt_encoder(
                        text=None,
                        points=pt,
                        boxes=None,
                        masks=None,
                    )
                    pred, _ = net.mask_decoder(
                        image_embeddings=imge,
                        image_pe=net.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=se,
                        dense_prompt_embeddings=de, 
                        multimask_output=False,
                    )
                # post process
                pred = F.interpolate(
                    pred,
                    (ori_masks.shape[2], ori_masks.shape[3]),
                    mode="bilinear",
                    align_corners=False,
                ) 
                sigmoid_pred = torch.sigmoid(pred)
                binary_pred = (sigmoid_pred > 0.5).float()
                binary_masks = (ori_masks > 0.5).float()
                dice_score_list = dice_coeff(binary_pred[:,0,:,:], binary_masks[:,0,:,:])    
                for i in range(len(pos_neg)):
                    if pos_neg[i] == 1:
                        pos_dice_list.append(dice_score_list[i])
                    elif pos_neg[i] == 0:
                        neg_dice_list.append(dice_score_list[i])
                    overall_dice_list.append(dice_score_list[i])
                if organ_print_record[texts[0]] > 0:
                    concatenated_pred = torch.cat((concatenated_pred, binary_pred[:,0,:,:]), dim=0)
                    concatenated_mask = torch.cat((concatenated_mask, binary_masks[:,0,:,:]), dim=0)
                    concatenated_imgs = torch.cat((concatenated_imgs, ori_imgs[0][0].permute(2, 0, 1)), dim=0)
                bottom += 64
            if organ_print_record[texts[0]] > 0:
                draw_cal_volume(concatenated_pred.permute(1, 2, 0), concatenated_mask.permute(1, 2, 0), concatenated_imgs.permute(1, 2, 0), batch_idx, epoch, 'eval', texts[0], overall_dice_list, concatenated_point)
                organ_print_record[texts[0]] -= 1
            organ_dice_list[texts[0]].extend(overall_dice_list)
            organ_pos_dice_list[texts[0]].extend(pos_dice_list)
            organ_neg_dice_list[texts[0]].extend(neg_dice_list)
            organ_dice[texts[0]].append(sum(overall_dice_list) / len(overall_dice_list))
            pbar.update() 
    return organ_dice_list, organ_pos_dice_list, organ_neg_dice_list, organ_dice

def eval_sam_btcv(args, eval_loader, epoch, net: nn.Module, clean_dir=True):
    # eval mode
    net.eval()

    n_val = len(eval_loader)  # the number of batch
    GPUdevice = torch.device('cuda:' + str(args.gpu_device))
    device = GPUdevice

    organ_dice_list = {
        "spleen" : [],
        "kidney right" : [],
        "kidney left" : [],
        "gallbladder" : [],
        "esophagus" : [],
        "liver" : [],
        "stomach" : [],
        "aorta" : [],
        "inferior vena cava" : [],
        "portal vein and splenic vein" : [],
        "pancreas" : [],
        "adrenal gland" : [],
    }
    organ_pos_dice_list = {
        "spleen" : [],
        "kidney right" : [],
        "kidney left" : [],
        "gallbladder" : [],
        "esophagus" : [],
        "liver" : [],
        "stomach" : [],
        "aorta" : [],
        "inferior vena cava" : [],
        "portal vein and splenic vein" : [],
        "pancreas" : [],
        "adrenal gland" : [],
    }
    organ_neg_dice_list = {
        "spleen" : [],
        "kidney right" : [],
        "kidney left" : [],
        "gallbladder" : [],
        "esophagus" : [],
        "liver" : [],
        "stomach" : [],
        "aorta" : [],
        "inferior vena cava" : [],
        "portal vein and splenic vein" : [],
        "pancreas" : [],
        "adrenal gland" : [],
    }
    organ_print_record = {
        "spleen" : 2,
        "kidney right" : 2,
        "kidney left" : 2,
        "gallbladder" : 2,
        "esophagus" : 2,
        "liver" : 2,
        "stomach" : 2,
        "aorta" : 2,
        "inferior vena cava" : 2,
        "portal vein and splenic vein" : 2,
        "pancreas" : 2,
        "adrenal gland" : 2,
    }
    organ_dice = {
        "spleen" : [],
        "kidney right" : [],
        "kidney left" : [],
        "gallbladder" : [],
        "esophagus" : [],
        "liver" : [],
        "stomach" : [],
        "aorta" : [],
        "inferior vena cava" : [],
        "portal vein and splenic vein" : [],
        "pancreas" : [],
        "adrenal gland" : [],
    }

    with tqdm(total=n_val, desc='evaluate round', unit='batch', leave=False) as pbar:
        for batch_idx, pack in enumerate(eval_loader):
            imgsw = pack['image'].to(dtype = torch.float32, device = GPUdevice)
            masksw = pack['label'].to(dtype = torch.float32, device = GPUdevice)
            texts = pack['text']
            depth = imgsw.shape[4]
            bottom = 0
            pos_dice_list = []
            neg_dice_list = []
            overall_dice_list = []
            concatenated_pred = torch.empty(0, masksw.shape[2], masksw.shape[3]).to(dtype = torch.float32, device = GPUdevice)
            concatenated_mask = torch.empty(0, masksw.shape[2], masksw.shape[3]).to(dtype = torch.float32, device = GPUdevice)
            concatenated_imgs = torch.empty(0, masksw.shape[2], masksw.shape[3]).to(dtype = torch.float32, device = GPUdevice)
            concatenated_point = torch.empty(0, 2).to(dtype = torch.float32, device = GPUdevice)
            while bottom < depth:
                output_shape = (256, 256, min(depth-bottom, 64))
                ori_imgs = imgsw[:,:,:,:,bottom:bottom + min(depth-bottom, 64)]
                imgs = F.interpolate(ori_imgs, size=output_shape, mode='trilinear', align_corners=False)
                ori_masks = masksw[:,:,:,:,bottom:bottom + min(depth-bottom, 64)]
                masks = F.interpolate(ori_masks, size=output_shape, mode='trilinear', align_corners=False)
                if 'pt' not in pack:
                    imgs, pt, masks, pos_neg, bboxes = generate_click_prompt(imgs, masks)  
                else:
                    pt = pack['pt']
                    point_labels = pack['p_label']

                if args.thd:
                    pt = rearrange(pt, 'b n d -> (b d) n')
                    pos_neg = rearrange(pos_neg, 'b d -> (b d)')
                    imgs = rearrange(imgs, 'b c h w d -> (b d) c h w ')
                    ori_masks = rearrange(ori_masks, 'b c h w d -> (b d) c h w ')
                    imgs = imgs.repeat(1,3,1,1)
                    point_labels = torch.ones(imgs.size(0))
                showp = pt
                concatenated_point = torch.cat((concatenated_point, showp), dim=0)
                if point_labels[0] != -1:
                    # point_coords = samtrans.ResizeLongestSide(longsize).apply_coords(pt, (h, w))
                    point_coords = pt
                    coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=GPUdevice)
                    labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=GPUdevice)
                    coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
                    pt = (coords_torch, labels_torch)
                
                '''test'''
                with torch.no_grad():
                    imge= net.image_encoder(imgs)
                    texts_aug = []
                    texts_aug.append('A computerized tomography of a '+texts[0])
                    text_tolken = clip.tokenize(texts).to(device)
                    texte = clip_model.encode_text(text_tolken)
                    texte = texte.unsqueeze(1)
                    texte = texte.to(torch.float32)
                    se, de = net.prompt_encoder(
                        text=None,
                        points=pt,
                        boxes=None,
                        masks=None,
                    )
                    pred, _ = net.mask_decoder(
                        image_embeddings=imge,
                        image_pe=net.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=se,
                        dense_prompt_embeddings=de, 
                        multimask_output=False,
                    )
                # post process
                pred = F.interpolate(
                    pred,
                    (ori_masks.shape[2], ori_masks.shape[3]),
                    mode="bilinear",
                    align_corners=False,
                ) 
                sigmoid_pred = torch.sigmoid(pred)
                binary_pred = (sigmoid_pred > 0.5).float()
                binary_masks = (ori_masks > 0.5).float()
                dice_score_list = dice_coeff(binary_pred[:,0,:,:], binary_masks[:,0,:,:])    
                for i in range(len(pos_neg)):
                    if pos_neg[i] == 1:
                        pos_dice_list.append(dice_score_list[i])
                    elif pos_neg[i] == 0:
                        neg_dice_list.append(dice_score_list[i])
                    overall_dice_list.append(dice_score_list[i])
                if organ_print_record[texts[0]] > 0:
                    concatenated_pred = torch.cat((concatenated_pred, binary_pred[:,0,:,:]), dim=0)
                    concatenated_mask = torch.cat((concatenated_mask, binary_masks[:,0,:,:]), dim=0)
                    concatenated_imgs = torch.cat((concatenated_imgs, ori_imgs[0][0].permute(2, 0, 1)), dim=0)
                bottom += 64
            # if organ_print_record[texts[0]] > 0:
            #     draw_cal_volume(concatenated_pred.permute(1, 2, 0), concatenated_mask.permute(1, 2, 0), concatenated_imgs.permute(1, 2, 0), batch_idx, epoch, 'eval_btcv', texts[0], overall_dice_list, concatenated_point)
            #     organ_print_record[texts[0]] -= 1
            organ_dice_list[texts[0]].extend(overall_dice_list)
            organ_pos_dice_list[texts[0]].extend(pos_dice_list)
            organ_neg_dice_list[texts[0]].extend(neg_dice_list)
            organ_dice[texts[0]].append(sum(overall_dice_list) / len(overall_dice_list))
            pbar.update() 
    return organ_dice_list, organ_pos_dice_list, organ_neg_dice_list, organ_dice

# def eval_sam(args, eval_loader, epoch, net: nn.Module, clean_dir=True):
#      # eval mode
#     net.eval()

#     mask_type = torch.float32
#     n_val = len(eval_loader)  # the number of batch
#     hard = 0
#     threshold = (0.1, 0.3, 0.5, 0.7, 0.9)
#     GPUdevice = torch.device('cuda:' + str(args.gpu_device))
#     device = GPUdevice
#     plabel = 1
#     patch_num_list = []

#     ct_min_norm = {
#         'Liver' : -2.96,
#         'Lung Cancer' : -2.67,
#         'Pancreas' : -2.31,
#         'Hepatic Vessels' : -2.04,
#         'Spleen' : -3.55,
#         'Colon Cancer' : -2.82,
#     }

#     with tqdm(total=n_val, desc='evaluate round', unit='batch', leave=False) as pbar:
#         pos_num = 0
#         pos_intersection_total = 0
#         pos_union_total = 0
#         total_dice_patch_list = []
#         neg_intersection_total = 0
#         neg_union_total = 0
#         neg_num = 0

#         for ind, pack in enumerate(eval_loader):
#             imgsw = pack['image'].to(dtype = torch.float32, device = GPUdevice)
#             masksw = pack['label'].to(dtype = torch.float32, device = GPUdevice)
#             texts = pack['text']
#             total_intersection_total = 0
#             total_union_total = 0

#             imgsw = imgsw.squeeze()
#             masksw = masksw.squeeze()
#             imgsw_np = imgsw.cpu().numpy()
#             masksw_np = masksw.cpu().numpy()
#             patch_size = (args.roi_size, args.roi_size, args.evl_chunk)
#             img_patch = patch(imgsw_np, patch_size, ct_min_norm[texts[0]])
#             mask_patch = patch(masksw_np, patch_size, 0)
#             imgsw = torch.tensor(img_patch)
#             masksw = torch.tensor(mask_patch)
#             patch_num_list.append(imgsw.shape[0])

#             for i in range(imgsw.shape[0]):
#                 imgs = imgsw[i]
#                 masks = masksw[i]
#                 imgs, pt, masks, pos_neg = generate_click_prompt(imgs, masks, plabel)
#                 if args.thd:
#                     pt = rearrange(pt, 'b n d -> (b d) n')
#                     pos_neg = rearrange(pos_neg, 'b d -> (b d)')
#                     imgs = rearrange(imgs, 'b c h w d -> (b d) c h w ')
#                     masks = rearrange(masks, 'b c h w d -> (b d) c h w ')
#                     imgs = imgs.repeat(1,3,1,1)
#                     point_labels = torch.ones(imgs.size(0))

#                 mask_type = torch.float32

#                 if point_labels[0] != -1:
#                     # point_coords = samtrans.ResizeLongestSide(longsize).apply_coords(pt, (h, w))
#                     point_coords = pt
#                     coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=GPUdevice)
#                     labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=GPUdevice)
#                     coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
#                     pt = (coords_torch, labels_torch)
                    
#                 imgs = imgs.to(dtype = mask_type,device = GPUdevice)
#                 masks = masks.to(dtype = mask_type,device = GPUdevice)

#                 '''test'''
#                 with torch.no_grad():
#                     imge= net.image_encoder(imgs)
#                     se, de = net.prompt_encoder(
#                         points=pt,
#                         boxes=None,
#                         masks=None,
#                     )
#                     text_tolken = clip.tokenize(texts).to(device)
#                     texte = clip_model.encode_text(text_tolken)
#                     texte = texte.unsqueeze(1)
#                     texte = texte.to(torch.float32)
#                     texte = text_prompt_embedding(texte)
#                     pred, _ = net.mask_decoder(
#                         image_embeddings=imge,
#                         image_pe=net.prompt_encoder.get_dense_pe(),
#                         sparse_prompt_embeddings=texte,
#                         dense_prompt_embeddings=de, 
#                         multimask_output=False,
#                     )
#                     # post process
#                     pred = F.interpolate(
#                         pred,
#                         (args.out_size, args.out_size),
#                         mode="bilinear",
#                         align_corners=False,
#                     ) 
#                     # comaper slice based 3d dice 
#                     # metric = DiceMetric(include_background=True, reduction="mean")
#                     # d1 = cal_3dslice_dice(pred, masks)
#                     # binary_y_true = (masks > 0.5).float()
#                     # sigmoid_pred = torch.sigmoid(pred)
#                     # binary_y_pred = (sigmoid_pred >= 0.5).float()
#                     # d2 = metric(binary_y_pred, binary_y_true)
#                     # print('compare', d1, d2)
#                     for i in  range(pos_neg.shape[0]) :
#                         intersection, union, _ = cal_intersection_union(pred[i], masks[i])
#                         total_intersection_total += intersection
#                         total_union_total += union
#                         if  pos_neg[i] == 0:
#                             neg_intersection_total += intersection
#                             neg_union_total += union
#                             neg_num += 1
#                         else :
#                             pos_intersection_total += intersection
#                             pos_union_total += union
#                             pos_num += 1
#             total_dice_patch = (2.0 * total_intersection_total) / (total_union_total + 1e-10)
#             total_dice_patch_list.append(total_dice_patch)
#             pbar.update()

#         neg_ratio = neg_num / (pos_num + neg_num)
#         pos_pf = (2.0 * pos_intersection_total) / (pos_union_total + 1e-10)
#         neg_pf = (neg_union_total / neg_num) / (args.roi_size * args.roi_size)
#         overall_dice = sum(total_dice_patch_list) / len(total_dice_patch_list)
#         print('neg / total', neg_ratio, pos_num, neg_num)  
#         print('pos_pf', pos_pf) 
#         print('neg_pf', neg_pf)
#         print('overall_dice', overall_dice) 
            
#     return overall_dice, pos_pf, neg_pf


# draw masks
    # if pos_print and first_pos_index >= 0 :
    #     sigmoid_pred = torch.sigmoid(pred)
    #     binary_pred = (sigmoid_pred >= args.threshold).float()    
    #     pred_mask = binary_pred[first_pos_index][0].detach().cpu().numpy()
    #     gt_mask = masks[first_pos_index][0].detach().cpu().numpy()
    #     img = imgs[first_pos_index][0].detach().cpu().numpy()
    #     point_loc = showp[first_pos_index].cpu().numpy()
    #     # 使用 Matplotlib 绘制图像
    #     plt.figure(figsize=(10, 3))  # 调整图像大小             

    #     # 绘制第一个张量的图像
    #     plt.subplot(131)  # 分割为 1x3 子图，选择第一个子图
    #     plt.imshow(pred_mask, cmap='gray')
    #     plt.title('pred_mask')               

    #     # 绘制第二个张量的图像
    #     plt.subplot(132)  # 选择第二个子图
    #     plt.imshow(gt_mask, cmap='gray')
    #     plt.scatter(point_loc[1], point_loc[0], color='red', marker='o')
    #     plt.title('gt_mask')               

    #     # 绘制第三个张量的图像
    #     plt.subplot(133)  # 选择第三个子图
    #     plt.imshow(img, cmap='gray')
    #     plt.scatter(point_loc[1], point_loc[0], color='red', marker='o')
    #     plt.title('img')               

    #     # 调整子图布局
    #     plt.tight_layout()              

    #     # 保存图像
    #     plt.savefig(f'image/train/Positive_{epoch}.png')
    #     plt.close()
    #     pos_print = False

    # if neg_print and first_neg_index >= 0 :
    #     sigmoid_pred = torch.sigmoid(pred)
    #     binary_pred = (sigmoid_pred >= args.threshold).float()    
    #     pred_mask = binary_pred[first_neg_index][0].detach().cpu().numpy()
    #     gt_mask = masks[first_neg_index][0].detach().cpu().numpy()
    #     img = imgs[first_neg_index][0].detach().cpu().numpy()
    #     point_loc = showp[first_neg_index].cpu().numpy()
    #     # 使用 Matplotlib 绘制图像
    #     plt.figure(figsize=(10, 3))  # 调整图像大小             

    #     # 绘制第一个张量的图像
    #     plt.subplot(131)  # 分割为 1x3 子图，选择第一个子图
    #     plt.imshow(pred_mask, cmap='gray')
    #     plt.title('pred_mask')               

    #     # 绘制第二个张量的图像
    #     plt.subplot(132)  # 选择第二个子图
    #     plt.imshow(gt_mask, cmap='gray')
    #     plt.scatter(point_loc[1], point_loc[0], color='red', marker='o')
    #     plt.title('gt_mask')               

    #     # 绘制第三个张量的图像
    #     plt.subplot(133)  # 选择第三个子图
    #     plt.imshow(img, cmap='gray')
    #     plt.scatter(point_loc[1], point_loc[0], color='red', marker='o')
    #     plt.title('img')               

    #     # 调整子图布局
    #     plt.tight_layout()              

    #     # 保存图像
    #     plt.savefig(f'image/train/Negative_{epoch}.png')
    #     plt.close()
    #     neg_print = False





    # for j in range(len(plabel_list)):
    #     plable_intersection_total = 0
    #     plable_union_total = 0
    #     for i in range(imgsw.shape[0]):
    #         imgs = imgsw[i]
    #         masks = masksw[i]
    #         imgs, pt, masks = generate_click_prompt(imgs, masks, plabel_list[j])
    #         if args.thd:
    #             pt = rearrange(pt, 'b n d -> (b d) n')
    #             imgs = rearrange(imgs, 'b c h w d -> (b d) c h w ')
    #             masks = rearrange(masks, 'b c h w d -> (b d) c h w ')
    #             imgs = imgs.repeat(1,3,1,1)
    #             point_labels = torch.ones(imgs.size(0))

    #         mask_type = torch.float32
    #         # ind += 1
    #         b_size,c,w,h = imgs.size()
    #         longsize = w if w >=h else h

    #         if point_labels[0] != -1:
    #             # point_coords = samtrans.ResizeLongestSide(longsize).apply_coords(pt, (h, w))
    #             point_coords = pt
    #             coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=GPUdevice)
    #             labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=GPUdevice)
    #             coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
    #             pt = (coords_torch, labels_torch)

    #         '''init'''
    #         if hard:
    #             true_mask_ave = (true_mask_ave > 0.5).float()
    #             #true_mask_ave = cons_tensor(true_mask_ave)
    #         imgs = imgs.to(dtype = mask_type,device = GPUdevice)
    #         masks = masks.to(dtype = mask_type,device = GPUdevice)

    #         '''test'''
    #         with torch.no_grad():
    #             imge= net.image_encoder(imgs)
    #             se, de = net.prompt_encoder(
    #                 points=pt,
    #                 boxes=None,
    #                 masks=None,
    #             )
    #             pred, _ = net.mask_decoder(
    #                 image_embeddings=imge,
    #                 image_pe=net.prompt_encoder.get_dense_pe(),
    #                 sparse_prompt_embeddings=se,
    #                 dense_prompt_embeddings=de, 
    #                 multimask_output=False,
    #             )
    #             # post process
    #             pred = F.interpolate(
    #                 pred,
    #                 (args.out_size, args.out_size),
    #                 mode="bilinear",
    #                 align_corners=False,
    #             ) 
                
    #             intersection, union = cal_intersection_union(pred, masks)
    #             plable_intersection_total += intersection
    #             plable_union_total += union

    #     dice_score = (2.0 * plable_intersection_total) / (plable_union_total + 1e-10)
    #     iou_score = plable_intersection_total / (plable_union_total - plable_intersection_total)
    #     plabel_dice_total[j] += dice_score
    #     plabel_iou_total[j] += iou_score


    # draw masks
    # if pos_print and first_pos_index >= 0 :
    #     sigmoid_pred = torch.sigmoid(pred)
    #     binary_pred = (sigmoid_pred >= args.threshold).float()    
    #     pred_mask = binary_pred[first_pos_index][0].detach().cpu().numpy()
    #     gt_mask = masks[first_pos_index][0].detach().cpu().numpy()
    #     img = imgs[first_pos_index][0].detach().cpu().numpy()
    #     point_loc = showp[first_pos_index].cpu().numpy()
    #     # 使用 Matplotlib 绘制图像
    #     plt.figure(figsize=(10, 3))  # 调整图像大小             

    #     # 绘制第一个张量的图像
    #     plt.subplot(131)  # 分割为 1x3 子图，选择第一个子图
    #     plt.imshow(pred_mask, cmap='gray')
    #     plt.title('pred_mask')               

    #     # 绘制第二个张量的图像
    #     plt.subplot(132)  # 选择第二个子图
    #     plt.imshow(gt_mask, cmap='gray')
    #     plt.scatter(point_loc[1], point_loc[0], color='red', marker='o')
    #     plt.title('gt_mask')               

    #     # 绘制第三个张量的图像
    #     plt.subplot(133)  # 选择第三个子图
    #     plt.imshow(img, cmap='gray')
    #     plt.scatter(point_loc[1], point_loc[0], color='red', marker='o')
    #     plt.title('img')               

    #     # 调整子图布局
    #     plt.tight_layout()              

    #     # 保存图像
    #     plt.savefig(f'image/valid/Positive_{epoch}.png')
    #     plt.close()
    #     pos_print = False

    # if neg_print and first_neg_index >= 0 :
    #     sigmoid_pred = torch.sigmoid(pred)
    #     binary_pred = (sigmoid_pred >= args.threshold).float()    
    #     pred_mask = binary_pred[first_neg_index][0].detach().cpu().numpy()
    #     gt_mask = masks[first_neg_index][0].detach().cpu().numpy()
    #     img = imgs[first_neg_index][0].detach().cpu().numpy()
    #     point_loc = showp[first_neg_index].cpu().numpy()
    #     # 使用 Matplotlib 绘制图像
    #     plt.figure(figsize=(10, 3))  # 调整图像大小             

    #     # 绘制第一个张量的图像
    #     plt.subplot(131)  # 分割为 1x3 子图，选择第一个子图
    #     plt.imshow(pred_mask, cmap='gray')
    #     plt.title('pred_mask')               

    #     # 绘制第二个张量的图像
    #     plt.subplot(132)  # 选择第二个子图
    #     plt.imshow(gt_mask, cmap='gray')
    #     plt.scatter(point_loc[1], point_loc[0], color='red', marker='o')
    #     plt.title('gt_mask')               

    #     # 绘制第三个张量的图像
    #     plt.subplot(133)  # 选择第三个子图
    #     plt.imshow(img, cmap='gray')
    #     plt.scatter(point_loc[1], point_loc[0], color='red', marker='o')
    #     plt.title('img')               

    #     # 调整子图布局
    #     plt.tight_layout()              

    #     # 保存图像
    #     plt.savefig(f'image/valid/Negative_{epoch}.png')
    #     plt.close()
    #     neg_print = False