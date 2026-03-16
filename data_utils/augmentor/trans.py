import math
import numbers
from typing import Tuple, List, Dict, Any

import cv2
import numpy as np
import random
import albumentations as A
import torch
from PIL import Image, ImageOps
from torchvision import transforms as T
from torchvision.transforms import functional as F
from scipy.ndimage import rotate


def pad_if_smaller(img, size, fill=0):
    # 如果图像最小边长小于给定size，则用数值fill进行padding
    min_size = min(img.size)
    if min_size < size:
        ow, oh = img.size
        padh = size - oh if oh < size else 0
        padw = size - ow if ow < size else 0
        img = F.pad(img, (0, 0, padw, padh), fill=fill)
    return img


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomResize(object):
    def __init__(self, min_size, max_size=None):
        self.min_size = min_size
        if max_size is None:
            max_size = min_size
        self.max_size = max_size

    def __call__(self, image, target):
        size = random.randint(self.min_size, self.max_size)
        # 这里size传入的是int类型，所以是将图像的最小边长缩放到size大小
        image = F.resize(image, size)
        # 这里的interpolation注意下，在torchvision(0.9.0)以后才有InterpolationMode.NEAREST
        # 如果是之前的版本需要使用PIL.Image.NEAREST
        target = F.resize(target, size, interpolation=T.InterpolationMode.NEAREST)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = F.hflip(Image.fromarray(image))
            target = F.hflip(Image.fromarray(target))
        return np.array(image), np.array(target)


class RandomVerticalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = F.vflip(Image.fromarray(image))
            target = F.vflip(Image.fromarray(target))
        return np.array(image), np.array(target)


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = pad_if_smaller(image, self.size)
        target = pad_if_smaller(target, self.size, fill=255)
        crop_params = T.RandomCrop.get_params(image, (self.size, self.size))
        image = F.crop(image, *crop_params)
        target = F.crop(target, *crop_params)
        return image, target


class RandomCrop2(object):
    def __init__(self, size):
        if isinstance(size, tuple) or isinstance(size, list):
            self.size = size  # 这个size是个元组
        else:
            self.size = (size, size)

    def __call__(self, image, target):
        flag = False
        if isinstance(image, np.ndarray):
            flag = True
            image = Image.fromarray(image)
            target = Image.fromarray(target)
        crop_params = T.RandomCrop.get_params(image, self.size)
        image = F.crop(image, *crop_params)
        target = F.crop(target, *crop_params)
        if flag:
            image = np.array(image)
            target = np.array(target)
        return image, target


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = F.center_crop(image, self.size)
        target = F.center_crop(target, self.size)
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)  # (C,H,W)
        target = torch.as_tensor(np.array(target), dtype=torch.int64)
        return image, target


class ToNumpy(object):
    """
    将PIL的数据转换为Numpy,以便处理
    """

    def __call__(self, image, target):
        rgb_image = np.array(image)
        target = np.array(target)
        return rgb_image, target


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target


class ConvertToGrayscale(object):
    def __init__(self, weights=None):
        if weights is None:
            weights = [0.299, 0.587, 0.114]
        self.weights = weights

    def __call__(self, image, target):
        """
        自定义RGB权重进行灰度化
        Args:
            image: 三通道图像
            weights: RGB每个通道的权重, 相加必须为1

        Returns:

        """
        # 计算RGB通道的加权平均值，并将其赋值给灰度图像的每个像素
        image_float = image.astype(np.float32)
        grayscale_image = np.dot(image_float[..., :3], self.weights).astype(np.uint8)

        # cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) 默认灰度, 权重为 [0.299, 0.587, 0.114]
        return grayscale_image, target


class Clahe(object):
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        # self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, image, target):
        """
        限制对比度自适应直方图均衡化
        """
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)

        if len(image.shape) == 3 and image.shape[-1] == 3:
            # print(image_data.shape)
            imgr = image[:, :, 0]
            imgg = image[:, :, 1]
            imgb = image[:, :, 2]

            cllr = clahe.apply(imgr)
            cllg = clahe.apply(imgg)
            cllb = clahe.apply(imgb)
            image_data = np.dstack((cllr, cllg, cllb))
            return image_data, target
        else:
            return clahe.apply(image.float()), target


class GammaCorrection(object):
    def __init__(self, gamma=1.0):
        self.gamma = gamma

    def __call__(self, image, target):
        """
        伽玛校正
        """
        invGamma = 1.0 / self.gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table), target


class ToPIL:
    def __call__(self, image, target):
        """
        灰度的 ndarray 转为 PIL.Image
        """
        return Image.fromarray(image), target


class AddGaussianNoise(object):

    def __init__(self, prob=0.2):
        self.prob = prob

    def __call__(self, image, target):
        """
        添加高斯噪音
        """
        trans = A.GaussNoise(p=self.prob)
        transformed = trans(image=image)
        return transformed["image"], target


class RandomReflectRotate(object):
    """
    带镜像填充的随机旋转, 避免引入黑色伪影
    """

    def __init__(self, angle=20, prob=0.5):
        self.angle = angle
        self.prob = prob

    def __call__(self, image, mask):
        if random.random() >= self.prob:
            return image, mask

        angle = random.uniform(-self.angle, self.angle)
        # 转换为Tensor
        img_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float()  # (C,H,W) [0-255]
        mask_tensor = torch.from_numpy(np.array(mask)).float()  # (H,W) [0-255]

        # 计算动态填充尺寸
        angle_rad = math.radians(abs(angle))
        max_dim = max(img_tensor.shape[1], img_tensor.shape[2])
        pad_size = int(math.ceil(max_dim * math.tan(angle_rad)))  # 增加1像素冗余

        # 反射填充
        img_padded = torch.nn.functional.pad(img_tensor, (pad_size,) * 4, mode='reflect')
        mask_padded = torch.nn.functional.pad(mask_tensor.unsqueeze(0), (pad_size,) * 4, mode='reflect')[0]

        # 执行旋转
        rotated_img = F.rotate(img_padded, angle,
                               interpolation=F.InterpolationMode.BILINEAR)
        rotated_mask = F.rotate(mask_padded.unsqueeze(0), angle,
                                interpolation=F.InterpolationMode.NEAREST)[0].long()

        # 裁剪回原尺寸
        h, w = img_tensor.shape[1], img_tensor.shape[2]
        rotated_img = rotated_img[:, pad_size:-pad_size, pad_size:-pad_size]
        rotated_mask = rotated_mask[pad_size:-pad_size, pad_size:-pad_size]

        # 转回Numpy
        return (
            rotated_img.permute(1, 2, 0).byte().numpy(),  # (H,W,C)
            rotated_mask.byte().numpy().astype(np.uint8)
        )


class PolarTransform(object):
    """
    极坐标变换
    """

    def __init__(self):
        pass

    def __call__(self, image, label):
        angle = -90
        # 获取图像尺寸
        h, w, c = image.shape
        center = (w // 2, h // 2)  # 假设ROI已经是中心裁剪的视盘区域
        max_radius = min(center[0], center[1])  # 最大变换半径

        # 图像极坐标变换（线性插值）
        img_polar = cv2.linearPolar(
            image,
            center,
            max_radius,
            cv2.WARP_FILL_OUTLIERS  # 最近邻保持类别
        )

        # 标签极坐标变换（最近邻插值）
        label_polar = cv2.linearPolar(
            label,
            center,
            max_radius,
            cv2.WARP_FILL_OUTLIERS  # 最近邻保持类别
        )

        # 旋转调整方向（可选步骤）
        img_rotated = rotate(img_polar, angle, reshape=False)
        label_rotated = rotate(label_polar, angle, reshape=False)
        return img_rotated, label_rotated


class Resize(object):
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def __call__(self, image: np.ndarray, label: np.ndarray):
        # 使用双线性插值缩放图像
        img_resize = cv2.resize(
            src=image,
            dsize=(self.width, self.height),  # (宽度, 高度)
            interpolation=cv2.INTER_LINEAR
        )

        # 使用最近邻插值缩放标签（避免产生无效的插值标签）
        label_resize = cv2.resize(
            src=label,
            dsize=(self.width, self.height),  # (宽度, 高度)
            interpolation=cv2.INTER_NEAREST
        )
        return img_resize, label_resize


class RandomCropAndPad(object):
    def __init__(self, prob=0.5, rand_scale=0.2):
        self.prob = prob
        self.rand_scale = rand_scale

    def __call__(self, image: np.ndarray, label: np.ndarray):
        trans = A.CropAndPad(percent=[-self.rand_scale, self.rand_scale], p=self.prob)
        transformed = trans(image=image, mask=label)
        img_A = transformed['image']
        mask_A = transformed['mask']
        return img_A, mask_A


class RandomColorJitter(object):
    def __init__(self):
        pass

    def __call__(self, image: np.ndarray, label: np.ndarray):
        image_aug_func = T.RandomChoice([
            T.ColorJitter(brightness=0.2),
            T.ColorJitter(contrast=0.2),
            T.ColorJitter(saturation=0.2),
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0),
        ])
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        return np.array(image_aug_func(image)), label


class nnUnetTransformer(object):
    def __init__(self):
        pass

    def __call__(self, image: np.ndarray, label: np.ndarray):
        """
        添加高斯噪音
        """
        trans = A.Compose([
            A.Affine(  # 大小变化, 变小边界填充0，变大裁剪中心，最终大小不变
                scale=(0.7, 1.4),
                translate_percent=(0, 0),
                rotate=(0, 0),
                shear=(0, 0),
                p=0.2
            ),
            A.Affine(  # x,y轴随机移动
                scale=(1, 1),
                translate_percent=(-0.05, 0.05),
                rotate=(0, 0),
                shear=(0, 0),
                p=0.2
            ),
            A.Affine(
                scale=(1, 1),
                translate_percent=(0, 0),
                rotate=(-45, 45),
                shear=(0, 0),
                p=0.2
            ),
            A.GaussNoise(
                std_range=(0, 0.1),
                mean_range=(0, 0),
                per_channel=True,
                p=0.1
            ),
            A.GaussianBlur(
                blur_limit=0,
                sigma_limit=(0.5, 1.0),
                p=0.2
            ),
            A.RandomBrightnessContrast(
                brightness_limit=(-0.25, 0.25),
                contrast_limit=(0, 0),
                brightness_by_max=True,
                ensure_safe_range=False,
                p=0.15
            ),
            A.RandomBrightnessContrast(
                brightness_limit=(0, 0),
                contrast_limit=(-0.25, 0.25),
                brightness_by_max=True,
                ensure_safe_range=False,
                p=0.15
            ),
            A.Downscale(
                scale_range=(0.5, 1),
                interpolation_pair={"upscale": 0, "downscale": 0},
                p=0.2
            ),
            A.RandomGamma(
                gamma_limit=(70, 150),
                p=0.1
            ),
            A.VerticalFlip(p=0.2),
            A.HorizontalFlip(p=0.2),
        ])
        result = trans(image=image, mask=label)
        return result['image'], result['mask']


class OCTA500Transformer(object):
    def __init__(self, crop_size: Tuple[int, int] = (100, 100)):
        """
        初始化转换器
        :param crop_size: 裁剪尺寸 (width, long)
        """
        self.crop_size = crop_size
        # 定义基础的2D变换
        self.base_transforms = A.ReplayCompose([
            A.Affine(
                scale=(1, 1),
                translate_percent=(0, 0),
                rotate=(-15, 15),
                shear=(0, 0),
                p=0.2
            ),
            # 可以添加更多Albumentations支持的2D变换
        ])

    def _random_crop(self, modal_list: List[np.ndarray], label: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        随机裁剪width和long维度，保持height不变
        """
        # 获取随机裁剪的起始点
        h = modal_list[0].shape[0]  # height保持不变
        w, l = modal_list[0].shape[1], modal_list[0].shape[2]  # width和long维度
        crop_w, crop_l = self.crop_size

        # 确保裁剪尺寸不超过原始尺寸
        if w < crop_w or l < crop_l:
            raise ValueError(f"Crop size {self.crop_size} larger than input size {(w, l)}")

        # 随机选择裁剪位置
        left = random.randint(0, w - crop_w)
        long_start = random.randint(0, l - crop_l)

        # 对所有模态数据进行裁剪
        cropped_modals = []
        for modal in modal_list:
            # 保持height不变，裁剪width和long维度
            cropped = modal[:, left:left + crop_w, long_start:long_start + crop_l]
            cropped_modals.append(cropped)

        # 对标签(width, long)进行相同的裁剪
        cropped_label = label[left:left + crop_w, long_start:long_start + crop_l]

        return cropped_modals, cropped_label

    def __call__(self, modal_list: List[np.ndarray], label: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        执行数据增强
        :param modal_list: 包含oct和octa数据的列表 [(height, width, long), (height, width, long)]
        :param label: 标签数据 (width, long)
        :return: 增强后的模态数据和标签
        """
        # 1. 随机裁剪width和long维度
        # cropped_modals, cropped_label = self._random_crop(modal_list, label)
        if len(self.base_transforms) == 0:
            return modal_list, label

        cropped_modals, cropped_label = modal_list, label
        h, w, l = cropped_modals[0].shape

        sample_slice = cropped_modals[0][0, :, :]  # 用第一个模态的第一个高度层来决定随机参数
        sampled = self.base_transforms(image=sample_slice, mask=cropped_label)
        replay = sampled["replay"]  # <- 正确：使用 data["replay"]
        transformed_label = sampled.get("mask", None)

        # 2) 用相同的 replay 对所有模态、所有高度层复用参数
        transformed_modals = []
        for modal in cropped_modals:
            transformed_modal = np.zeros_like(modal)
            for height_idx in range(h):
                slice_data = modal[height_idx, :, :]
                # 将采样到的参数应用到每一层（注意调用方式）
                out = A.ReplayCompose.replay(replay, image=slice_data)
                transformed_modal[height_idx, :, :] = out["image"]
            transformed_modals.append(transformed_modal)

        return transformed_modals, transformed_label


class OCTA5002DTransformer(object):
    def __init__(self, crop_size: Tuple[int, int] = (100, 100)):
        """
        初始化转换器
        :param crop_size: 裁剪尺寸 (width, long)
        """
        self.crop_size = crop_size
        # 定义基础的2D变换
        self.base_transforms = A.Compose([
            A.Affine(  # 大小变化, 变小边界填充0，变大裁剪中心，最终大小不变
                scale=(0.7, 1.4),
                translate_percent=(0, 0),
                rotate=(0, 0),
                shear=(0, 0),
                p=0.2
            ),
            A.Affine(  # x,y轴随机移动
                scale=(1, 1),
                translate_percent=(-0.05, 0.05),
                rotate=(0, 0),
                shear=(0, 0),
                p=0.2
            ),
            A.Affine(
                scale=(1, 1),
                translate_percent=(0, 0),
                rotate=(-45, 45),
                shear=(0, 0),
                p=0.2
            ),
            A.GaussNoise(
                std_range=(0, 0.1),
                mean_range=(0, 0),
                per_channel=True,
                p=0.1
            ),
            A.GaussianBlur(
                blur_limit=0,
                sigma_limit=(0.5, 1.0),
                p=0.2
            ),
            A.RandomBrightnessContrast(
                brightness_limit=(-0.25, 0.25),
                contrast_limit=(0, 0),
                brightness_by_max=True,
                ensure_safe_range=False,
                p=0.15
            ),
            A.RandomBrightnessContrast(
                brightness_limit=(0, 0),
                contrast_limit=(-0.25, 0.25),
                brightness_by_max=True,
                ensure_safe_range=False,
                p=0.15
            ),
            A.Downscale(
                scale_range=(0.5, 1),
                interpolation_pair={"upscale": 0, "downscale": 0},
                p=0.2
            ),
            A.RandomGamma(
                gamma_limit=(70, 150),
                p=0.1
            ),
            A.VerticalFlip(p=0.2),
            A.HorizontalFlip(p=0.2),
        ])

    def __call__(self, image, label: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        添加高斯噪音
        """
        result = self.base_transforms(image=image, mask=label)
        return result['image'], result['mask']


if __name__ == '__main__':
    t = AddGaussianNoise()
    img = np.random.randn(224, 224, 3)
    mask = np.random.randint(1, 255, size=(224, 224))
    img, mask = t(img, mask)
    print(img, mask)
