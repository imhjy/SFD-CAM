import json
import os
from PIL import Image
import numpy as np
from sklearn.model_selection import KFold
from torch.utils.data import Dataset

from utils.common import replace_system_separator


def find_center_of_mass(label_array):
    # 合并视杯(0)和视盘(128)为目标区域（排除背景255）
    mask = (label_array != 0)
    labeled_pixels = np.argwhere(mask)
    if len(labeled_pixels) == 0:
        raise ValueError("未找到有效视杯视盘区域")
    center_y, center_x = np.mean(labeled_pixels, axis=0)
    return int(round(center_x)), int(round(center_y))  # 四舍五入更精准[3](@ref)


def crop_roi(image, center_x, center_y, roi_size):
    """
    :param image: Pillow图像对象
    :param center_x: 质心X坐标
    :param center_y: 质心Y坐标
    :param roi_size: 目标尺寸（width, height）
    """
    width, height = image.size
    crop_w, crop_h = roi_size, roi_size

    # 计算裁剪边界
    left = max(0, center_x - crop_w // 2)
    top = max(0, center_y - crop_h // 2)
    right = min(width, left + crop_w)
    bottom = min(height, top + crop_h)

    return image.crop((left, top, right, bottom))


class OrigaDataset(Dataset):
    """
    数据集: ORIGA
    """

    def __init__(self, hypes, train: bool, transforms=None, fold=0):
        super(OrigaDataset, self).__init__()
        self.transforms = transforms
        root = hypes['dataset']['root_dir']
        self.train_expand_rate = hypes['dataset']['train_expand_rate']
        self.roi_size = hypes['dataset']['roi_size']
        self.train = train
        if fold == 0:
            data_root = root
            img_names = [i for i in os.listdir(os.path.join(data_root, "train")) if i.endswith(".jpg")]
            self.img_list = [os.path.join(data_root, "train", i) for i in img_names]
            self.label = [os.path.join(data_root, "masks_train", i.split('.')[0] + '.png')
                          for i in img_names]
            # check files
            for i in self.label:
                if os.path.exists(i) is False:
                    raise FileNotFoundError(f"file {i} does not exists.")
        else:
            # 使用fold
            json_path = os.path.join(root, 'fold.json')
            with open(json_path, 'r', encoding='utf-8') as f:
                json_dict = json.load(f)
            flag = "train" if train else "test"
            self.img_list = [os.path.join(root, ab_path) for ab_path in json_dict[f'fold-{fold}'][f'{flag}_images']]
            self.label = [os.path.join(root, ab_path) for ab_path in json_dict[f'fold-{fold}'][f'{flag}_label']]

    def __getitem__(self, idx):
        if self.train:
            idx = idx % len(self.img_list)
        # 1. 读取图像和标签
        img = Image.open(replace_system_separator(self.img_list[idx])).convert('RGB')
        label = Image.open(replace_system_separator(self.label[idx])).convert('L')  # 图像转换为灰度

        # 2. 找到ROI区域(EYE-OCD跳过)
        center_x, center_y = find_center_of_mass(np.array(label))
        img = crop_roi(img, center_x, center_y, self.roi_size)
        label = crop_roi(label, center_x, center_y, self.roi_size)

        # 3. 转换标签
        label = np.array(label)
        mask = np.zeros_like(label, dtype=np.uint8)

        # 标记区域(背景)
        mask[label == 0] = 0

        # 标记区域(视盘)
        mask[label == 1] = 1

        # 标记区域(视杯)
        mask[label == 2] = 2

        # 这里转回PIL的原因是，transforms中是对PIL数据进行处理
        mask = Image.fromarray(mask)

        if self.transforms is not None:
            img, mask = self.transforms(img, mask)

        return img, mask

    def __len__(self):
        if self.train:
            return len(self.img_list) * self.train_expand_rate
        return len(self.img_list)

    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)
        return batched_imgs, batched_targets

    @staticmethod
    def fold_dataset_split(hypes):
        """
        切分数据集, 默认使用五折交叉检验, 每个dataset都需要实现这个方法
        Args:
            hypes: 配置文件
        """

        def get_image_number(dataset_path):
            train_path = os.path.join(dataset_path, 'train')
            test_path = os.path.join(dataset_path, 'test')
            return len(os.listdir(train_path) + os.listdir(test_path))

        def get_image_path_list(dataset_path):
            train_path = os.path.join(dataset_path, 'train')
            test_path = os.path.join(dataset_path, 'test')
            images_list = np.array(
                [os.path.join('train', file) for file in os.listdir(train_path)] +
                [os.path.join('test', file) for file in os.listdir(test_path)]

            )
            return images_list

        def get_label_path_list(images_list):
            label_list = []

            for path in images_list:
                file_name = os.path.basename(path)  # 获取文件名（含后缀）
                file_name = os.path.splitext(file_name)[0]  # 去除后缀
                if path.find("train") != -1:
                    label_list.append(os.path.join('masks_train',f'{file_name}.png'))
                else:
                    label_list.append(os.path.join('masks_test',f'{file_name}.png'))

            return np.array(label_list)

        fold_num = hypes['dataset']['fold_num']
        kf = KFold(n_splits=fold_num, shuffle=False)  # 初始化KFold
        dataset_path = hypes['dataset']['root_dir']

        image_size = get_image_number(dataset_path)
        idx_list = [i for i in range(image_size)]
        # 得到相对位置
        images_list = get_image_path_list(dataset_path)

        label_list = get_label_path_list(images_list)

        json_dict = {}
        for index, (train_index, test_index) in enumerate(kf.split(idx_list)):  # 调用split方法切分数据
            json_dict[f'fold-{index + 1}'] = {}
            json_dict[f'fold-{index + 1}']['train_images'] = list(images_list[train_index])
            json_dict[f'fold-{index + 1}']['train_label'] = list(label_list[train_index])

            json_dict[f'fold-{index + 1}']['test_images'] = list(images_list[test_index])
            json_dict[f'fold-{index + 1}']['test_label'] = list(label_list[test_index])
            print('train_index:%s , test_index: %s ' % (train_index, test_index))
        file_txt = json.dumps(json_dict)
        with open(os.path.join(dataset_path, 'fold.json'), 'w', encoding='utf-8') as f:
            f.write(file_txt)


def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs
