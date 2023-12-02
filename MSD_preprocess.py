import os
import json
import torch
from monai.transforms import (
    AsDiscrete,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    EnsureTyped,
    SpatialCrop,
    AddChanneld,
    Transform,
    ResizeWithPadOrCropd,
    Lambda,
)
from monai.data import (
    ThreadDataLoader,
    CacheDataset,
    load_decathlon_datalist,
    decollate_batch,
    set_track_meta,
    DataLoader,
)

import pickle
import numpy as np
device = torch.device('cuda', 0)


def get_train_transform():
    return Compose(
        [   
            LoadImaged(keys=["image", "label"], ensure_channel_first=None),
            AddChanneld(keys=["image", "label"]),
            # ScaleIntensityRanged(
            #     keys=["image"],
            #     a_min=-1000,
            #     a_max=350,
            #     b_min=0.0,
            #     b_max=1.0,
            #     clip=True,
            # ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.5, 1.5, 1.5),
                mode=("bilinear", "nearest"),
            ),
            EnsureTyped(keys=["image", "label"], device=device, track_meta=False),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                # spatial_size=(patch[0],patch[1],patch[2]),
                spatial_size=(None,None,64),
                allow_smaller=True,
                pos=1,
                neg=0,
                num_samples=1,
                image_key="image",
                image_threshold=0,
            ),
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[0],
                prob=0.10,
            ),
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[1],
                prob=0.10,
            ),
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[2],
                prob=0.10,
            ),
            RandRotate90d(
                keys=["image", "label"],
                prob=0.10,
                max_k=3,
            ),
            RandShiftIntensityd(
                keys=["image"],
                offsets=0.10,
                prob=0.50,
            ),
        ]
    )

def get_evaluation_transform():
    return Compose(
        [
            LoadImaged(keys=["image", "label"], ensure_channel_first=None),
            AddChanneld(keys=["image", "label"]),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-1000,
                a_max=350,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            Spacingd(
                keys=["image", "label"],
                # pixdim=(spacing[0], spacing[1], spacing[2]),
                pixdim=(1.5, 1.5, 1.5),
                mode=("bilinear", "nearest"),
            ),
            EnsureTyped(keys=["image", "label"], device=device, track_meta=True),
        ]
    )

def custom_load_decathlon_datalist(json_data_path, index):
    with open(json_data_path, 'r', encoding='utf-8') as json_file:
        json_data = json_file.read()
    parsed_data = json.loads(json_data)
    target = parsed_data[index]
    for i in range(len(target)):
        target[i]['image'] = data_dir + target[i]['image'][1:]
        target[i]['label'] = data_dir + target[i]['label'][1:]

    return target

patch_size = {
    3: [64, 64, 64],
    6: [64, 64, 64],
    7: [64, 64, 64],
    8: [64, 64, 64],
    9: [64, 64, 64],
    10: [64, 64, 64],
}

spacings = {
    2: [1.25, 1.25, 1.37],
    3: [0.7676, 0.7676, 1],
    6: [0.79, 0.79, 1.24],
    7: [0.8, 0.8, 2.5],
    8: [0.8, 0.8, 1.5],
    9: [0.78, 0.78, 1.6],
    10: [0.78, 0.78, 3],
}

ct_mean = {
    3: 99.4, 
    6: -158.58, 
    7: 77.9, 
    8: 104.37, 
    9: 99.29, 
    10: 62.18,
}

ct_std = {
    3: 39.36, 
    6: 324.7, 
    7: 75.4, 
    8: 52.62, 
    9: 39.47, 
    10: 32.65,
}

ct_min = {
    3: -17,
    6: -1024,
    7: -96,
    8: -3,
    9: -41,
    10: -30,
}

ct_max = {
    3: 201,
    6: 325,
    7: 215,
    8: 243,
    9: 176,
    10: 165.82,
}

organ_text = {
    3: 'Liver',
    6: 'Lung Cancer',
    7: 'Pancreas',
    8: 'Hepatic Vessels',
    9: 'Spleen',
    10: 'Colon Cancer',
}
# A computerized tomography of a 
data_dir = './data'
task_list = [3,6,7,8,9,10] #
imgs = []
labels = []
text = []
organ_num = {
    3:131, #70
    6:63, #32
    7:281, #139
    8:303, #140
    9:41, #20
    10:126, #64
}
for task in task_list:
    split_JSON = f"dataset_{task}.json"
    datasets = os.path.join(data_dir, split_JSON)
    datalist = custom_load_decathlon_datalist(datasets, "training") 
    # split_index = organ_num[task]* 4 // 5
    # segment1 = datalist[0:7]
    # segment2 = datalist[split_index:min(split_index+7, organ_num[task])]
    # 连接两个段
    # eval_list = segment1 + segment2
    # print('eval_list', eval_list, len(eval_list))
    # datalist = datalist[:min(len(datalist), 200)]
    dataset = CacheDataset(
        data = datalist,
        transform = get_train_transform(),
        cache_num = 24,
        cache_rate = 1.0,
        num_workers = 8,
    )   

    # 遍历数据集并将结果存储为NumPy数组
    for i in range(len(dataset)):
        for j in range(len(dataset[i])):
            item = dataset[i][j]
            image = item["image"].cpu().numpy()
            label = item["label"].cpu().numpy()
            label[label != 1] = 0
            clipped_image = np.clip(image, ct_min[task], ct_max[task])
            z_score_normalized_data = (clipped_image - ct_mean[task]) / ct_std[task]
            imgs.append(z_score_normalized_data)
            labels.append(label)
            text.append(organ_text[task])
            
            # 使用 pickle.dump 将列表存储到文件中
            # np.save(f'./numpy_data_slice/train_vol/large_overall/imgs{index}', image)

            # # 使用 pickle.dump 将列表存储到文件中
            # np.save(f'./numpy_data_slice/train_vol/large_overall/labels{index}', label)

            # for j in range(len(item)):
        #     image = item[j]["image"].cpu().numpy()
        #     label = item[j]["label"].cpu().numpy()
        #     print('check', image.shape, label.shape)
        #     label[label != 1] = 0
        #     # clipped_image = np.clip(image, ct_min[task], ct_max[task])
        #     # z_score_normalized_data = (clipped_image - ct_mean[task]) / ct_std[task]
        #     imgs.append(image)
        #     labels.append(label)
        #     text.append(organ_text[task])

print('results', len(imgs), len(labels), len(text))
print('text', text)
# 使用 pickle.dump 将列表存储到文件中

with open(f'./numpy_data_slice/train_vol/large_pos_overall/imgs.pkl', 'wb') as file:
    pickle.dump(imgs, file)

with open(f'./numpy_data_slice/train_vol/large_pos_overall/labels.pkl', 'wb') as file:
    pickle.dump(labels, file)

with open(f'./numpy_data_slice/train_vol/large_pos_overall/texts.pkl', 'wb') as file:
    pickle.dump(text, file)

# data_dir = './data'
# task_list = [3,6,7,8,9,10] #3,6,7,8,,10
# imgs = []
# labels = []
# text = []
# organ_num = {
#     3:131, #70
#     6:63, #32
#     7:281, #139
#     8:303, #140
#     9:41, #20
#     10:126, #64
# }
# for task in task_list:
#     split_JSON = f"dataset_{task}.json"
#     datasets = os.path.join(data_dir, split_JSON)
#     datalist = custom_load_decathlon_datalist(datasets, "training") 
#     split_index = organ_num[task]* 4 // 5
#     datalist = datalist[split_index:]

#     dataset = CacheDataset(
#         data = datalist,
#         transform = get_evaluation_transform(),
#         cache_num = 24,
#         cache_rate = 1.0,
#         num_workers = 8,
#     )   

#     # split_index = len(datalist) * 4 // 5

#     # 遍历数据集并将结果存储为NumPy数组
#     for i in range(len(datalist)):
#         # print('i', i)
#         item = dataset[i]
#         image = item["image"].cpu().numpy()
#         label = item["label"].cpu().numpy()
#         label[label != 1] = 0
#         # clipped_image = np.clip(image, ct_min[task], ct_max[task])
#         # z_score_normalized_data = (clipped_image - ct_mean[task]) / ct_std[task]
#         imgs.append(image)
#         labels.append(label)
#         text.append(organ_text[task])
#         # for j in range(len(item)):
#         #     image = item[j]["image"].cpu().numpy()
#         #     label = item[j]["label"].cpu().numpy()
#         #     print('check', image.shape, label.shape)
#         #     label[label != 1] = 0
#         #     # clipped_image = np.clip(image, ct_min[task], ct_max[task])
#         #     # z_score_normalized_data = (clipped_image - ct_mean[task]) / ct_std[task]
#         #     imgs.append(image)
#         #     labels.append(label)
#         #     text.append(organ_text[task])

# print('results', len(imgs), len(labels), len(text))
# print('text', text)


# # 使用 pickle.dump 将列表存储到文件中
# with open('./numpy_data_slice/eval/overall_small/imgs.pkl', 'wb') as file:
#     pickle.dump(imgs, file)

# # 使用 pickle.dump 将列表存储到文件中
# with open('./numpy_data_slice/eval/overall_small/labels.pkl', 'wb') as file:
#     pickle.dump(labels, file)

# # 使用 pickle.dump 将列表存储到文件中
# with open('./numpy_data_slice/eval/overall_small/texts.pkl', 'wb') as file:
#     pickle.dump(text, file)


