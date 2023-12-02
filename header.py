import os
import monai
import nibabel as nib
import json
import numpy as np
from tqdm import tqdm
from monai.transforms import Compose, LoadImaged, AddChanneld, EnsureChannelFirst, CropForegroundd
import torch

data_dir = '../data'
class_list = 8 # 2,3,6,7,8,9,10

transform = Compose([
    LoadImaged(keys=["image", "label"], ensure_channel_first=None),
    AddChanneld(keys=["image", "label"]),
    CropForegroundd(keys=["image", "label"], source_key="image"),  # 設定你希望的 "foreground crop" 範圍
])

def custom_load_decathlon_datalist(json_data_path, index):
    with open(json_data_path, 'r', encoding='utf-8') as json_file:
        json_data = json_file.read()
    parsed_data = json.loads(json_data)
    target = parsed_data[index]
    for i in range(len(target)):
        target[i]['image'] = data_dir+ target[i]['image'][1:]
        target[i]['label'] = data_dir + target[i]['label'][1:]

    return target


split_JSON = f"dataset_{class_list}.json"
datasets = os.path.join(data_dir, split_JSON)
datalist = custom_load_decathlon_datalist(datasets, "training")
pixdim_list = [[], [], []]
dim_list = [[], [], []]
vmin = []
vmax = []
xrange_list = []
yrange_list = []
zrange_list = []

xrange_list2 = []
yrange_list2 = []
zrange_list2 = []

with open(f'task{class_list}.txt', 'w') as file:
    for i in tqdm(range(len(datalist))) : 
        # 設定 .nii.gz 檔案的路徑
        file_path = datalist[i]['image']  
        label_path = datalist[i]['label']  

        # 使用 nibabel 讀取 .nii.gz 檔案
        img = nib.load(file_path)  
        label = nib.load(label_path)  

        # data = transform(datalist[i])['image']
        # foreground_voxels = torch.squeeze(data, dim=0)
        # print('success', img.get_fdata().shape, data.shape)
        # print('VALUE', np.min(img.get_fdata()), np.max(img.get_fdata()), torch.min(data), torch.max(data))
        # data = cropped_img.get_fdata()
        
        # 選擇前景像素（非背景像素），這裡假設背景值為0
        foreground_voxels = img.get_fdata()[label.get_fdata() != 0]
        print('foreground_voxels', len(foreground_voxels))
        # 計算0.5和99.5百分位數
        percentile_0_5 = np.percentile(foreground_voxels, 0.5)
        percentile_99_5 = np.percentile(foreground_voxels, 99.5)
        print(f"value: {percentile_0_5}, {percentile_99_5}")
        
        vmin.append(percentile_0_5)
        vmax.append(percentile_99_5)


        # 取得像素或格點大小
        pixdim = img.header["pixdim"][1:4]  # 前四個元素中的第二到第四個是像素或格點大小
        dim = img.header["dim"][1:4]  # 前四個元素中的第二到第四個是像素或格點大小  

        # # 印出像素或格點大小
        # print(f"spacing (mm): {pixdim}")
        # print(f"dim: {dim}")
        # for i in range(3) :
        #     pixdim_list[i].append(pixdim[i])
        #     dim_list[i].append(dim[i])

        # file.write(f"{file_path[17:]} spacing (mm): {pixdim}")
        # file.write(f" dim: {dim}\n")

        label = label.get_fdata()
        target_indices = np.where(label == 1)
        min_coords = np.min(target_indices, axis=1)
        max_coords = np.max(target_indices, axis=1)
        x_range = (max_coords[0] - min_coords[0]) * pixdim[0]
        y_range = (max_coords[1] - min_coords[1]) * pixdim[1]
        z_range = (max_coords[2] - min_coords[2]) * pixdim[2]
        print('range', x_range, y_range, z_range)
        xrange_list.append(x_range)
        yrange_list.append(y_range)
        zrange_list.append(z_range)


        target_indices = np.where(label == 2)
        if target_indices[0].size > 0:
            min_coords = np.min(target_indices, axis=1)
            max_coords = np.max(target_indices, axis=1)
            x_range = (max_coords[0] - min_coords[0]) * pixdim[0]
            y_range = (max_coords[1] - min_coords[1]) * pixdim[1]
            z_range = (max_coords[2] - min_coords[2]) * pixdim[2]
            print('range', x_range, y_range, z_range)
            xrange_list2.append(x_range)
            yrange_list2.append(y_range)
            zrange_list2.append(z_range)
    
    # file.write(f"------------------------------\n")

    # data = pixdim_list[0]
    # median = np.median(data)
    # mean = np.mean(data)
    # maximum = np.max(data)
    # minimum = np.min(data)

    # file.write(f"spacing x => median {median}, mean {mean}, maximum {maximum}, minimum {minimum}\n")

    # data = pixdim_list[1]
    # median = np.median(data)
    # mean = np.mean(data)
    # maximum = np.max(data)
    # minimum = np.min(data)

    # file.write(f"spacing y => median {median}, mean {mean}, maximum {maximum}, minimum {minimum}\n")

    # data = pixdim_list[2]
    # median = np.median(data)
    # mean = np.mean(data)
    # maximum = np.max(data)
    # minimum = np.min(data)

    # file.write(f"spacing z => median {median}, mean {mean}, maximum {maximum}, minimum {minimum}\n")
    
    # data = dim_list[2]
    # median = np.median(data)
    # mean = np.mean(data)
    # maximum = np.max(data)
    # minimum = np.min(data)

    # file.write(f"depth => median {median}, mean {mean}, maximum {maximum}, minimum {minimum}\n")

    data = vmin
    median = np.median(data)
    mean = np.mean(data)
    maximum = np.max(data)
    minimum = np.min(data)

    file.write(f"vmin => median {median}, mean {mean}, maximum {maximum}, minimum {minimum}\n")

    data = vmax
    median = np.median(data)
    mean = np.mean(data)
    maximum = np.max(data)
    minimum = np.min(data)

    file.write(f"vmax => median {median}, mean {mean}, maximum {maximum}, minimum {minimum}\n")

    data = xrange_list
    median = np.median(data)
    mean = np.mean(data)
    maximum = np.max(data)
    minimum = np.min(data)

    file.write(f"xrange => median {median}, mean {mean}, maximum {maximum}, minimum {minimum}\n")

    data = yrange_list
    median = np.median(data)
    mean = np.mean(data)
    maximum = np.max(data)
    minimum = np.min(data)

    file.write(f"yrange => median {median}, mean {mean}, maximum {maximum}, minimum {minimum}\n")
    
    data = zrange_list
    median = np.median(data)
    mean = np.mean(data)
    maximum = np.max(data)
    minimum = np.min(data)

    file.write(f"zrange => median {median}, mean {mean}, maximum {maximum}, minimum {minimum}\n")


    data = xrange_list2
    median = np.median(data)
    mean = np.mean(data)
    maximum = np.max(data)
    minimum = np.min(data)

    file.write(f"xrange2 => median {median}, mean {mean}, maximum {maximum}, minimum {minimum}\n")

    data = yrange_list2
    median = np.median(data)
    mean = np.mean(data)
    maximum = np.max(data)
    minimum = np.min(data)

    file.write(f"yrange2 => median {median}, mean {mean}, maximum {maximum}, minimum {minimum}\n")
    
    data = zrange_list2
    median = np.median(data)
    mean = np.mean(data)
    maximum = np.max(data)
    minimum = np.min(data)

    file.write(f"zrange2 => median {median}, mean {mean}, maximum {maximum}, minimum {minimum}\n")

