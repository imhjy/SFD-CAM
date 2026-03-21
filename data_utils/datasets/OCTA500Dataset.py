import json
import os

import h5pickle
import torch
from PIL import Image
import numpy as np
from sklearn.model_selection import KFold
from torch.utils.data import Dataset
import natsort
from utils.common import replace_system_separator
import imageio.v2 as imageio
import h5py
from PIL import Image
import random


class OCTA500Dataset(Dataset):
    """
    数据集:OCTA-500
    这是一个用于OCTA-500数据集的自定义数据集类，继承自torch.utils.data.Dataset。
    该类提供了数据加载、预处理、数据集划分等功能，支持训练和测试模式。
    """

    def __init__(self, hypes, train: bool, transforms=None, fold=0):

        """
        初始化数据集
        Args:
            hypes: 配置文件，包含数据集路径、参数等信息
            train: 是否为训练模式
            transforms: 数据增强变换
            fold: 交叉验证的折数，0表示不使用交叉验证
        """
        super(OCTA500Dataset, self).__init__()  # 注意：这里父类名与实际继承的类名不一致
        self.transforms = transforms
        self.hypes = hypes
        self.root: str = hypes['dataset']['root_dir']  # 数据集根目录
        self.type: str = hypes['dataset']['type']
        self.input_type: str = hypes['input_type']
        self.modal_name: str = hypes['modal_name']
        self.train_expand_rate = hypes['dataset']['train_expand_rate']  # 训练数据扩展率
        self.train = train  # 训练/测试标志

        if self.input_type == '3d':
            save_root = os.path.join(self.root, self.type, 'processed')
            if not os.path.exists(os.path.join(save_root, "data.hdf5")):
                print('数据处理未完成，无法加载数据集')
                return
            self.data = {}
            f = h5pickle.File(os.path.join(save_root, "data.hdf5"), "r")
            self.data['Images'] = f['Images']
            self.data['Label'] = f['Label']
            # f.close()
            # 2. 检查是否已经分好数据集了，如果没有重新划分数据集
            json_path = os.path.join(os.path.join(self.root, self.type), 'fold.json')
            if not os.path.exists(json_path):
                print('请加载fold.json文件，再初始化数据集')
                return
            # 3. 获取数据集
            with open(json_path, 'r', encoding='utf-8') as f:
                json_dict = json.load(f)
            flag = "train" if train else "test"
            self.img_oct_list = [os.path.join(os.path.join(self.root, self.type), ab_path) for ab_path in
                                 json_dict[f'fold-{fold}'][f'{flag}_oct_images']]

            self.img_octa_list = [os.path.join(os.path.join(self.root, self.type), ab_path) for ab_path in
                                  json_dict[f'fold-{fold}'][f'{flag}_octa_images']]

            self.label = [os.path.join(os.path.join(self.root, self.type), ab_path) for ab_path in
                          json_dict[f'fold-{fold}'][f'{flag}_label']]
        elif self.input_type == '2d':
            json_path = os.path.join(os.path.join(self.root, self.type), 'fold.json')
            if not os.path.exists(json_path):
                print('正在进行数据集划分')
                OCTA500Dataset.fold_dataset_split(hypes)
            # 获取数据集
            with open(json_path, 'r', encoding='utf-8') as f:
                json_dict = json.load(f)
            flag = "train" if train else "test"
            self.img_list = [os.path.join(self.root, ab_path) for ab_path in
                             json_dict[f'fold-{fold}'][f'{flag}_2d_images']]
            self.label = [os.path.join(self.root, self.type, ab_path) for ab_path in
                          json_dict[f'fold-{fold}'][f'{flag}_label']]

    def __getitem__(self, idx):
        if self.train:
            idx = idx % len(self.img_oct_list if self.input_type == '3d' else self.img_list)
        if self.input_type == '3d':
            # 先找到idx和fold之间的对应关系
            abs_path = self.img_octa_list[idx]
            # 获取编号
            file_id = int(os.path.basename(abs_path))

            # 根据编号计算在self.data中的索引
            if 10001 <= file_id <= 10300:  # 6mm数据
                data_idx = file_id - 10001
            elif 10301 <= file_id <= 10500:  # 3mm数据
                data_idx = file_id - 10301  # 300是6mm数据的数量
            else:
                raise ValueError(f"Invalid file ID: {file_id}, should be between 10001-10500")

            # 1. 读取图像和标签(在这里就得把裁剪工作做了，不然太慢了)

            w, l = self.hypes['dataset']['data_size'][1], self.hypes['dataset']['data_size'][2]  # width和long维度
            crop_w, crop_l = self.hypes['dataset']['block_size'][1], self.hypes['dataset']['block_size'][2]

            # 随机选择裁剪位置
            left = random.randint(0, w - crop_w)
            long_start = random.randint(0, l - crop_l)

            input_data = []
            if self.train:
                if self.modal_name == 'ALL':
                    temp_data = self.data['Images'][:, :, left:left + crop_w, long_start:long_start + crop_l,
                                data_idx] #.astype(np.float32)
                    img_oct = temp_data[0, :, :, :]
                    img_octa = temp_data[1, :, :, :]
                    input_data = [img_oct, img_octa]
                elif self.modal_name == 'OCTA':
                    img_octa = self.data['Images'][1, :, left:left + crop_w, long_start:long_start + crop_l,
                               data_idx] #.astype(np.float32)
                    input_data = [img_octa]
                elif self.modal_name == 'OCT':
                    img_oct = self.data['Images'][0, :, left:left + crop_w, long_start:long_start + crop_l,
                              data_idx] #.astype(np.float32)
                    input_data = [img_oct]
                label = self.data['Label'][0, left:left + crop_w, long_start:long_start + crop_l, data_idx]
            else:
                if self.modal_name == 'ALL':
                    temp_data = self.data['Images'][:, :, :, :, data_idx] #.astype(np.float32)
                    img_oct = temp_data[0, :, :, :]
                    img_octa = temp_data[1, :, :, :]
                    input_data = [img_oct, img_octa]
                elif self.modal_name == 'OCTA':
                    img_octa = self.data['Images'][1, :, :, :, data_idx] #.astype(np.float32)
                    input_data = [img_octa]
                elif self.modal_name == 'OCT':
                    img_oct = self.data['Images'][0, :, :, :, data_idx] #.astype(np.float32)
                    input_data = [img_oct]
                label = self.data['Label'][0, :, :, data_idx]

            # 3. 转换标签
            label = np.array(label)
            mask = np.squeeze(label)

            # 假设经过测试得到的灰度值对应关系：
            # 白色 -> 255（对应原始RGB白色）
            # 红色 -> 76（由RGB红色[255,0,0]转换而来，0.299 * 255 ≈ 76）
            # 黑色 -> 0

            # 标记黑色区域（灰度值0-20）
            # mask[(label >= 0) & (label <= 50)] = 0

            # 标记红色区域（灰度值21-150）
            # mask[(label > 50) & (label <= 150)] = 1

            # 标记白色区域（灰度值151-255）
            # mask[label > 150] = 2

            img_processed_list, mask = self.transforms(input_data, mask)

            return img_processed_list, mask
        elif self.input_type == '2d':
            img = imageio.imread(replace_system_separator(self.img_list[idx]))
            label = np.array(Image.open(replace_system_separator(self.label[idx])).rotate(
                -90 if self.type != '6mm' else 90))  # 图像转换为灰度
            if self.train:
                w, l = self.hypes['dataset']['data_size'][1], self.hypes['dataset']['data_size'][2]  # width和long维度
                crop_w, crop_l = self.hypes['dataset']['block_size'][1], self.hypes['dataset']['block_size'][2]

                # 随机选择裁剪位置
                left = random.randint(0, w - crop_w)
                long_start = random.randint(0, l - crop_l)
                img = img[left:left + crop_w, long_start:long_start + crop_l].astype(np.float32)
                label = label[left:left + crop_w, long_start:long_start + crop_l]
            else:
                img = img.astype(np.float32)

            image_2d = None
            mask = None
            if self.transforms is not None:
                image_2d, mask = self.transforms(img, label)

            return image_2d, mask

    def __len__(self):
        if self.train and self.input_type == '3d':
            return len(self.img_octa_list) * self.train_expand_rate
        if not self.train and self.input_type == '3d':
            return len(self.img_octa_list)
        if self.train and self.input_type == '2d':
            return len(self.img_list) * self.train_expand_rate
        if not self.train and self.input_type == '2d':
            return len(self.img_list)
        return 0

    @staticmethod
    def collate_fn(batch):
        # 解包
        modal_lists, targets = zip(*batch)

        # 判断第一个样本的模态数量
        num_modals = len(modal_lists[0])

        # 处理 target
        batched_targets = cat_list(targets, fill_value=255)

        if num_modals == 1:
            # 单模态情况
            imgs = [m[0] for m in modal_lists]
            batched_imgs = cat_list(imgs, fill_value=0)
            return batched_imgs.unsqueeze(1), batched_targets

        elif num_modals == 2:
            # 双模态情况
            imgs1 = [m[0] for m in modal_lists]
            imgs2 = [m[1] for m in modal_lists]
            batched_imgs1 = cat_list(imgs1, fill_value=0)
            batched_imgs2 = cat_list(imgs2, fill_value=0)
            return torch.stack([batched_imgs1, batched_imgs2], dim=1), batched_targets

        else:
            raise ValueError(f"Unsupported modal count: {num_modals}")

    @staticmethod
    def collate_fn_2d(batch):
        images_2d, targets = list(zip(*batch))
        batched_imgs_2d = cat_list(images_2d, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)
        return batched_imgs_2d, batched_targets


def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs
