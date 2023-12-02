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

### Spleen
![image](https://github.com/SamuelWu2001/Universal-3D-Medical-Image-Segmentation-Model/assets/71746159/0a34f81b-1a28-45f1-a0c8-434728c3a7bf)
![36](https://github.com/SamuelWu2001/Universal-3D-Medical-Image-Segmentation-Model/assets/71746159/384e055c-2e4d-412a-814a-23100d0019bc)



