# Universal-3D-Medical-Image-Segmentation-Model
This project employs Parameter-Efficient Fine-Tuning methods to fine-tune the Segment Anything Model for implementing 3D medical image segmentation.

# Completed tasks
- Using PyTorch's checkpoint module to reduce the GPU memory required for training, allowing the entire model to train on a single NVIDIA RTX4090 (24GB) GPU.
- Implementing 3D random rotations and flips as transformations to augment the diversity of the dataset.
- Implementing preprocessing for the Medical Segmentation Decathlon to reduce GPU memory requirements and speed up training.
- Added adapter blocks to the mask decoder to implement Parameter-Efficient Fine-Tuning (PEFT).

# Result
![loss_plot](https://github.com/SamuelWu2001/Universal-3D-Medical-Image-Segmentation-Model/assets/71746159/52373d6c-d84f-4739-a9e9-6f77c93ad31d)

| | Liver (131) | Lung Cancer (63) | Pancreas (281) | Hepatic Vessels (303) | Spleen (41) | Colon Cancer (126)|
| -------- | -------- | -------- | -------- | -------- | -------- | -------- |
| SOTA |	95.42 | 80.01 | 82.84 | 67.15 | 97.27 | 63.14 |
| point_transform_small | 91.31 | 95.32 | 78.93 |	57.66 | 94.35 | 91.18 |

## Spleen
![36](https://github.com/SamuelWu2001/Universal-3D-Medical-Image-Segmentation-Model/assets/71746159/384e055c-2e4d-412a-814a-23100d0019bc)
![62](https://github.com/SamuelWu2001/Universal-3D-Medical-Image-Segmentation-Model/assets/71746159/393c3285-4670-4f54-bfca-f8f39305ccb8)

## Pancreas
![44](https://github.com/SamuelWu2001/Universal-3D-Medical-Image-Segmentation-Model/assets/71746159/673df7ba-6e33-4282-88ab-1ddce345eb04)
![19](https://github.com/SamuelWu2001/Universal-3D-Medical-Image-Segmentation-Model/assets/71746159/b896dc4a-941e-4c98-80e1-ce86645d60df)

## Lung Cancer
![54](https://github.com/SamuelWu2001/Universal-3D-Medical-Image-Segmentation-Model/assets/71746159/90072738-128d-44fe-85e3-b7cffdb7643d)
![33](https://github.com/SamuelWu2001/Universal-3D-Medical-Image-Segmentation-Model/assets/71746159/a3d3adf9-ac8c-4272-a17b-a357fc236327)

## Liver
![4](https://github.com/SamuelWu2001/Universal-3D-Medical-Image-Segmentation-Model/assets/71746159/56a6ca95-c6d8-4e5f-8e60-978b0978cb24)
![43](https://github.com/SamuelWu2001/Universal-3D-Medical-Image-Segmentation-Model/assets/71746159/7fdc8d5a-d67e-4686-ba32-be439c2fba4d)

## Hepatic Vessels
![56](https://github.com/SamuelWu2001/Universal-3D-Medical-Image-Segmentation-Model/assets/71746159/17dc3fcf-e621-4399-870b-dd95464fdff6)
![20](https://github.com/SamuelWu2001/Universal-3D-Medical-Image-Segmentation-Model/assets/71746159/18e46e74-ce38-45ee-a6bf-da9851f1f070)

## Colon Cancer
![35](https://github.com/SamuelWu2001/Universal-3D-Medical-Image-Segmentation-Model/assets/71746159/e3f016e0-19e2-4561-a991-9c3181f7fdc4)
![31](https://github.com/SamuelWu2001/Universal-3D-Medical-Image-Segmentation-Model/assets/71746159/241385b6-4129-46bb-a364-40c9ec6c30be)

# Reference
[Medical SAM Adapter: Adapting Segment Anything Model for Medical Image Segmentation](https://arxiv.org/abs/2304.12620)
