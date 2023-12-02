import numpy as np
from skimage.util import view_as_blocks

# # 创建一个示例 3D 数组，假设其形状为 (256, 256, 256)
# input_data = np.random.random((300, 300, 300))

# # 定义 patch 的大小
# patch_size = (64, 64, 64)
# # 计算需要的填充量
# pad_width = [
#     (0, patch_size[0] - input_data.shape[0] % patch_size[0]),
#     (0, patch_size[1] - input_data.shape[1] % patch_size[1]),
#     (0, patch_size[2] - input_data.shape[2] % patch_size[2])
# ]

# # 对输入数据进行填充
# padded_data = np.pad(input_data, pad_width, constant_values=1)
# # 切分 3D 数组为 patch
# patches = view_as_blocks(padded_data, patch_size)
# print(type(patches), patches.shape)
# # 打印切分后的 patch 数量和形状
# print("Number of Patches:", patches.shape[0] * patches.shape[1] * patches.shape[2])
# print("Patch Shape:", patches[0, 0, 0].shape)
# print('input_data', padded_data)
# print('patch1', patches[0, 0, 0])
# print('patch2', patches[0, 0, 4])

def patch(input_data, patch_size, pad_value):
    pad_width_x = (0, 0)
    pad_width_y = (0, 0)
    pad_width_z = (0, 0)
    # pad_width = [
    #     (0, patch_size[0] - input_data.shape[0] % patch_size[0]),
    #     (0, patch_size[1] - input_data.shape[1] % patch_size[1]),
    #     (0, patch_size[2] - input_data.shape[2] % patch_size[2])
    # ]

    if input_data.shape[0] % patch_size[0] > 0:
        pad_width_x = (0, patch_size[0] - input_data.shape[0] % patch_size[0])
    if input_data.shape[1] % patch_size[1] > 0:
        pad_width_y = (0, patch_size[1] - input_data.shape[1] % patch_size[1])
    if input_data.shape[2] % patch_size[2] > 0:
        pad_width_z = (0, patch_size[2] - input_data.shape[2] % patch_size[2])

    pad_width = [pad_width_x, pad_width_y, pad_width_z]
    padded_data = np.pad(input_data, pad_width, constant_values=pad_value)
    patches = view_as_blocks(padded_data, patch_size)
    patch_num = patches.shape[0] * patches.shape[1] * patches.shape[2]
    final_patch = patches.reshape((patch_num, 1, 1, patches.shape[3], patches.shape[4], patches.shape[5]))
    return final_patch

# input_data = [[[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0], [0,0,0,0,0,0], [0,0,0,0,0,0], [0,0,0,0,0,0]],
#               [[1,1,1,1,1,1],[1,1,1,1,1,1],[1,1,1,1,1,1], [1,1,1,1,1,1], [1,1,1,1,1,1], [1,1,1,1,1,1]],
#               [[2,2,2,2,2,2],[2,2,2,2,2,2],[2,2,2,2,2,2], [2,2,2,2,2,2], [2,2,2,2,2,2], [2,2,2,2,2,2]],
#               [[3,3,3,3,3,3],[3,3,3,3,3,3],[3,3,3,3,3,3], [3,3,3,3,3,3], [3,3,3,3,3,3], [3,3,3,3,3,3]],
#               [[4,4,4,4,4,4],[4,4,4,4,4,4],[4,4,4,4,4,4], [4,4,4,4,4,4], [4,4,4,4,4,4], [4,4,4,4,4,4]],
#               [[5,5,5,5,5,5],[5,5,5,5,5,5],[5,5,5,5,5,5], [5,5,5,5,5,5], [5,5,5,5,5,5], [5,5,5,5,5,5]]]
# a = np.array(input_data)
# r = patch(a, (2,2,2), -1)
# print(r.shape, r[0:9])
