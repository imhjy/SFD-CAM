import random

import numpy as np
import torch
from torchvision.transforms import functional as F

import data_utils.augmentor.trans as T


def check_foreground_area(target, threshold=0.05):
    """
    我们保证前景区域大于 threshold 才认为这是一个合格的样本
    Args:
        threshold: 阈值
        target:
    Returns:
    """
    pixels_mask = ((target > 0) & (target < 255)).float()

    # 计算值为255的像素总数
    pixels_num = pixels_mask.sum()

    # 计算总像素数
    total_num = target.numel()

    # 计算前景区域的比例
    return (pixels_num / total_num).item() >= threshold


class EnhanceGrayPresetTrain:
    def __init__(self, crop_size, hflip_prob=0.5, vflip_prob=0.5, gauss_prob=0.2, rotate_prob=0.5, angle=20,
                 mean=[0.38145922], std=[0.07928617], threshold=0.05, weights=[0.299, 0.587, 0.114], **kwargs):
        self.threshold = threshold
        trans = [T.RandomCrop2(crop_size),
                 T.ToNumpy()]
        # 随机高斯模糊 可能会降低精度
        # trans.append(T.AddGaussianNoise(prob=gauss_prob))
        # 随机旋转
        trans.append(T.RandomReflectRotate(angle=angle, prob=rotate_prob))
        trans.extend([
            T.ConvertToGrayscale(weights=weights),
            T.Clahe(),
            T.GammaCorrection(),
            T.ToPIL()])
        if hflip_prob > 0:
            trans.append(T.RandomHorizontalFlip(hflip_prob))
        if vflip_prob > 0:
            trans.append(T.RandomVerticalFlip(vflip_prob))
        trans.extend([
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        while True:
            trans_img, trans_target = self.transforms(img, target)
            if check_foreground_area(trans_target, self.threshold):
                break
        return trans_img, trans_target


class EnhanceGrayPresetEval:
    def __init__(self, mean=[0.38145922], std=[0.07928617], **kwargs):
        self.transforms = T.Compose([
            T.ToNumpy(),
            T.ConvertToGrayscale(),
            T.Clahe(),
            T.GammaCorrection(),
            T.ToPIL(),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)


class PolarPresetTrain:
    def __init__(self, hflip_prob=0.5, vflip_prob=0.5, rotate_prob=0.5, angle=20,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), calc_size=480, **kwargs):
        trans = [
            T.ToNumpy(),
            T.RandomCrop2(calc_size),
            T.AddGaussianNoise(),
            T.RandomColorJitter(),
            T.RandomCropAndPad(),
            # 随机旋转
            T.RandomReflectRotate(angle=angle, prob=rotate_prob),
            T.Clahe(),
            # 伽玛校正
            T.GammaCorrection(),
            T.RandomHorizontalFlip(hflip_prob),
            T.RandomVerticalFlip(vflip_prob),
            # 极坐标变换
            T.PolarTransform(),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        trans_img, trans_target = self.transforms(img, target)
        return trans_img, trans_target


class PolarPresetEval:
    def __init__(self, hflip_prob=0.5, vflip_prob=0.5, rotate_prob=0.5, angle=20,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), **kwargs):
        trans = [
            T.ToNumpy(),
            # 随机高斯模糊 可能会降低精度
            # trans.append(T.AddGaussianNoise(prob=gauss_prob))
            # 随机旋转
            # T.RandomReflectRotate(angle=angle, prob=rotate_prob),
            T.Clahe(),
            # 伽玛校正
            T.GammaCorrection(),
            # T.RandomHorizontalFlip(hflip_prob),
            # T.RandomVerticalFlip(vflip_prob),
            # 极坐标变换
            T.PolarTransform(),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        trans_img, trans_target = self.transforms(img, target)
        return trans_img, trans_target


class PolarPresetTrain2:
    def __init__(self, hflip_prob=0.5, vflip_prob=0.5, rotate_prob=0.5, angle=20,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), calc_size=480, **kwargs):
        trans = [
            T.ToNumpy(),
            # T.RandomCrop2(calc_size),
            T.AddGaussianNoise(),
            T.RandomColorJitter(),
            T.RandomCropAndPad(),
            # 随机旋转
            T.RandomReflectRotate(angle=angle, prob=rotate_prob),
            T.Clahe(),
            # 伽玛校正
            T.GammaCorrection(),
            T.RandomHorizontalFlip(hflip_prob),
            T.RandomVerticalFlip(vflip_prob),
            # 极坐标变换
            T.PolarTransform(),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        trans_img, trans_target = self.transforms(img, target)
        return trans_img, trans_target


class PolarPresetEval2:
    def __init__(self, hflip_prob=0.5, vflip_prob=0.5, rotate_prob=0.5, angle=20,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), **kwargs):
        trans = [
            T.ToNumpy(),
            # 随机高斯模糊 可能会降低精度
            # trans.append(T.AddGaussianNoise(prob=gauss_prob))
            # 随机旋转
            # T.RandomReflectRotate(angle=angle, prob=rotate_prob),
            T.Clahe(),
            # 伽玛校正
            T.GammaCorrection(),
            # T.RandomHorizontalFlip(hflip_prob),
            # T.RandomVerticalFlip(vflip_prob),
            # 极坐标变换
            T.PolarTransform(),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        trans_img, trans_target = self.transforms(img, target)
        return trans_img, trans_target


class PolarPresetTrain3:
    def __init__(self, hflip_prob=0.5, vflip_prob=0.5, rotate_prob=0.5, angle=20,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), calc_size=480, **kwargs):
        trans = [
            T.ToNumpy(),
            T.RandomCrop2(calc_size),
            # T.AddGaussianNoise(),
            T.RandomColorJitter(),
            T.RandomCropAndPad(),
            # 随机旋转
            T.RandomReflectRotate(angle=angle, prob=rotate_prob),
            T.Clahe(),
            # 伽玛校正
            T.GammaCorrection(),
            T.RandomHorizontalFlip(hflip_prob),
            T.RandomVerticalFlip(vflip_prob),
            # 极坐标变换
            T.PolarTransform(),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        trans_img, trans_target = self.transforms(img, target)
        return trans_img, trans_target


class PolarPresetEval3:
    def __init__(self, hflip_prob=0.5, vflip_prob=0.5, rotate_prob=0.5, angle=20,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), **kwargs):
        trans = [
            T.ToNumpy(),
            # 随机高斯模糊 可能会降低精度
            # trans.append(T.AddGaussianNoise(prob=gauss_prob))
            # 随机旋转
            # T.RandomReflectRotate(angle=angle, prob=rotate_prob),
            T.Clahe(),
            # 伽玛校正
            T.GammaCorrection(),
            # T.RandomHorizontalFlip(hflip_prob),
            # T.RandomVerticalFlip(vflip_prob),
            # 极坐标变换
            T.PolarTransform(),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        trans_img, trans_target = self.transforms(img, target)
        return trans_img, trans_target


class PolarPresetTrain4:
    def __init__(self, hflip_prob=0.5, vflip_prob=0.5, rotate_prob=0.5, angle=20,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), calc_size=480, **kwargs):
        trans = [
            T.ToNumpy(),
            T.RandomCrop2(calc_size),
            T.AddGaussianNoise(),
            # T.RandomColorJitter(),
            T.RandomCropAndPad(),
            # 随机旋转
            T.RandomReflectRotate(angle=angle, prob=rotate_prob),
            T.Clahe(),
            # 伽玛校正
            T.GammaCorrection(),
            T.RandomHorizontalFlip(hflip_prob),
            T.RandomVerticalFlip(vflip_prob),
            # 极坐标变换
            T.PolarTransform(),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        trans_img, trans_target = self.transforms(img, target)
        return trans_img, trans_target


class PolarPresetEval4:
    def __init__(self, hflip_prob=0.5, vflip_prob=0.5, rotate_prob=0.5, angle=20,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), **kwargs):
        trans = [
            T.ToNumpy(),
            # 随机高斯模糊 可能会降低精度
            # trans.append(T.AddGaussianNoise(prob=gauss_prob))
            # 随机旋转
            # T.RandomReflectRotate(angle=angle, prob=rotate_prob),
            T.Clahe(),
            # 伽玛校正
            T.GammaCorrection(),
            # T.RandomHorizontalFlip(hflip_prob),
            # T.RandomVerticalFlip(vflip_prob),
            # 极坐标变换
            T.PolarTransform(),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        trans_img, trans_target = self.transforms(img, target)
        return trans_img, trans_target


class PolarPresetTrain5:
    def __init__(self, hflip_prob=0.5, vflip_prob=0.5, rotate_prob=0.5, angle=20,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), calc_size=480, **kwargs):
        trans = [
            T.ToNumpy(),
            T.RandomCrop2(calc_size),
            T.AddGaussianNoise(),
            T.RandomColorJitter(),
            # T.RandomCropAndPad(),
            # 随机旋转
            T.RandomReflectRotate(angle=angle, prob=rotate_prob),
            T.Clahe(),
            # 伽玛校正
            T.GammaCorrection(),
            T.RandomHorizontalFlip(hflip_prob),
            T.RandomVerticalFlip(vflip_prob),
            # 极坐标变换
            T.PolarTransform(),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        trans_img, trans_target = self.transforms(img, target)
        return trans_img, trans_target


class PolarPresetEval5:
    def __init__(self, hflip_prob=0.5, vflip_prob=0.5, rotate_prob=0.5, angle=20,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), **kwargs):
        trans = [
            T.ToNumpy(),
            # 随机高斯模糊 可能会降低精度
            # trans.append(T.AddGaussianNoise(prob=gauss_prob))
            # 随机旋转
            # T.RandomReflectRotate(angle=angle, prob=rotate_prob),
            T.Clahe(),
            # 伽玛校正
            T.GammaCorrection(),
            # T.RandomHorizontalFlip(hflip_prob),
            # T.RandomVerticalFlip(vflip_prob),
            # 极坐标变换
            T.PolarTransform(),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        trans_img, trans_target = self.transforms(img, target)
        return trans_img, trans_target


class CommonPresetTrain:
    def __init__(self, hflip_prob=0.5, vflip_prob=0.5, rotate_prob=0.5, angle=20,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), calc_size=480, **kwargs):
        trans = [
            T.ToNumpy(),
            T.RandomCrop2(calc_size),
            T.AddGaussianNoise(),
            T.RandomColorJitter(),
            T.RandomCropAndPad(),
            # 随机旋转
            T.RandomReflectRotate(angle=angle, prob=rotate_prob),
            T.Clahe(),
            # 伽玛校正
            T.GammaCorrection(),
            T.RandomHorizontalFlip(hflip_prob),
            T.RandomVerticalFlip(vflip_prob),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        trans_img, trans_target = self.transforms(img, target)
        return trans_img, trans_target


class CommonPresetEval:
    def __init__(self, hflip_prob=0.5, vflip_prob=0.5, rotate_prob=0.5, angle=20,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), **kwargs):
        trans = [
            T.ToNumpy(),
            # 随机高斯模糊 可能会降低精度
            # trans.append(T.AddGaussianNoise(prob=gauss_prob))
            # 随机旋转
            # T.RandomReflectRotate(angle=angle, prob=rotate_prob),
            T.Clahe(),
            # 伽玛校正
            T.GammaCorrection(),
            # T.RandomHorizontalFlip(hflip_prob),
            # T.RandomVerticalFlip(vflip_prob),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        trans_img, trans_target = self.transforms(img, target)
        return trans_img, trans_target


class CommonPresetTrain2:
    def __init__(self, hflip_prob=0.5, vflip_prob=0.5, rotate_prob=0.5, angle=20,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), calc_size=480, **kwargs):
        trans = [
            T.ToNumpy(),
            T.AddGaussianNoise(),
            T.RandomColorJitter(),
            T.RandomCropAndPad(),
            # 随机旋转
            T.RandomReflectRotate(angle=angle, prob=rotate_prob),
            T.Clahe(),
            # 伽玛校正
            T.GammaCorrection(),
            T.RandomHorizontalFlip(hflip_prob),
            T.RandomVerticalFlip(vflip_prob),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        trans_img, trans_target = self.transforms(img, target)
        return trans_img, trans_target


class CommonPresetEval2:
    def __init__(self, hflip_prob=0.5, vflip_prob=0.5, rotate_prob=0.5, angle=20,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), **kwargs):
        trans = [
            T.ToNumpy(),
            T.Clahe(),
            # 伽玛校正
            T.GammaCorrection(),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        trans_img, trans_target = self.transforms(img, target)
        return trans_img, trans_target


class CommonPresetTrain3:
    def __init__(self, hflip_prob=0.5, vflip_prob=0.5, rotate_prob=0.5, angle=20,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), calc_size=480, **kwargs):
        trans = [
            T.ToNumpy(),
            T.RandomCrop2(calc_size),
            # T.AddGaussianNoise(),
            T.RandomColorJitter(),
            T.RandomCropAndPad(),
            # 随机旋转
            T.RandomReflectRotate(angle=angle, prob=rotate_prob),
            T.Clahe(),
            # 伽玛校正
            T.GammaCorrection(),
            T.RandomHorizontalFlip(hflip_prob),
            T.RandomVerticalFlip(vflip_prob),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        trans_img, trans_target = self.transforms(img, target)
        return trans_img, trans_target


class CommonPresetEval3:
    def __init__(self, hflip_prob=0.5, vflip_prob=0.5, rotate_prob=0.5, angle=20,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), **kwargs):
        trans = [
            T.ToNumpy(),
            T.Clahe(),
            # 伽玛校正
            T.GammaCorrection(),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        trans_img, trans_target = self.transforms(img, target)
        return trans_img, trans_target


class CommonPresetTrain4:
    def __init__(self, hflip_prob=0.5, vflip_prob=0.5, rotate_prob=0.5, angle=20,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), calc_size=480, **kwargs):
        trans = [
            T.ToNumpy(),
            T.RandomCrop2(calc_size),
            T.AddGaussianNoise(),
            # T.RandomColorJitter(),
            T.RandomCropAndPad(),
            # 随机旋转
            T.RandomReflectRotate(angle=angle, prob=rotate_prob),
            T.Clahe(),
            # 伽玛校正
            T.GammaCorrection(),
            T.RandomHorizontalFlip(hflip_prob),
            T.RandomVerticalFlip(vflip_prob),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        trans_img, trans_target = self.transforms(img, target)
        return trans_img, trans_target


class CommonPresetEval4:
    def __init__(self, hflip_prob=0.5, vflip_prob=0.5, rotate_prob=0.5, angle=20,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), **kwargs):
        trans = [
            T.ToNumpy(),
            T.Clahe(),
            # 伽玛校正
            T.GammaCorrection(),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        trans_img, trans_target = self.transforms(img, target)
        return trans_img, trans_target


class CommonPresetTrain5:
    def __init__(self, hflip_prob=0.5, vflip_prob=0.5, rotate_prob=0.5, angle=20,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), calc_size=480, **kwargs):
        trans = [
            T.ToNumpy(),
            T.RandomCrop2(calc_size),
            T.AddGaussianNoise(),
            T.RandomColorJitter(),
            # T.RandomCropAndPad(),
            # 随机旋转
            T.RandomReflectRotate(angle=angle, prob=rotate_prob),
            T.Clahe(),
            # 伽玛校正
            T.GammaCorrection(),
            T.RandomHorizontalFlip(hflip_prob),
            T.RandomVerticalFlip(vflip_prob),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        trans_img, trans_target = self.transforms(img, target)
        return trans_img, trans_target


class CommonPresetEval5:
    def __init__(self, hflip_prob=0.5, vflip_prob=0.5, rotate_prob=0.5, angle=20,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), **kwargs):
        trans = [
            T.ToNumpy(),
            T.Clahe(),
            # 伽玛校正
            T.GammaCorrection(),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        trans_img, trans_target = self.transforms(img, target)
        return trans_img, trans_target


class nnUnetPresetTrain:
    def __init__(self, hflip_prob=0.5, vflip_prob=0.5, rotate_prob=0.5, angle=20,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), calc_size=480, **kwargs):
        trans = [
            T.ToNumpy(),
            T.nnUnetTransformer(),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        trans_img, trans_target = self.transforms(img, target)
        return trans_img, trans_target


class nnUnetPresetEval:
    def __init__(self, hflip_prob=0.5, vflip_prob=0.5, rotate_prob=0.5, angle=20,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), **kwargs):
        trans = [
            T.ToNumpy(),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        trans_img, trans_target = self.transforms(img, target)
        return trans_img, trans_target


class OCTA500PresetTrain:
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), **kwargs):
        self.transforms = T.OCTA500Transformer()

    def __call__(self, modal_list, target):
        modal_list, target = self.transforms(modal_list, target)
        modal_list = [F.to_tensor(np.moveaxis(m, 0, -1)) for m in modal_list]
        # modal_list = [torch.from_numpy(modal_list[0]), torch.from_numpy(modal_list[1])]
        target = torch.as_tensor(np.array(target), dtype=torch.int64)
        # target = torch.from_numpy(np.array(target, dtype='long'))
        return modal_list, target


class OCTA500PresetEval:
    def __init__(self, **kwargs):
        # self.transforms = T.OCTA500TransformerEval()
        pass

    def __call__(self, modal_list, target):
        modal_list = [F.to_tensor(np.moveaxis(m, 0, -1)) for m in modal_list]
        # modal_list = [torch.from_numpy(modal_list[0]), torch.from_numpy(modal_list[1])]
        target = torch.as_tensor(np.array(target), dtype=torch.int64)
        # target = torch.from_numpy(np.array(target, dtype='long'))
        return modal_list, target

class OCTA5002DPresetTrain:
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), **kwargs):
        self.transforms = T.OCTA5002DTransformer()

    def __call__(self, image, target):
        image, target = self.transforms(image, target)
        # modal_list = [F.to_tensor(np.moveaxis(modal_list[0], 0, -1)), F.to_tensor(np.moveaxis(modal_list[1], 0, -1))]
        image = torch.from_numpy(image)
        # target = torch.as_tensor(np.array(target), dtype=torch.int64)
        target = torch.from_numpy(np.array(target, dtype='long'))
        return image, target


class OCTA5002DPresetEval:
    def __init__(self, **kwargs):
        # self.transforms = T.OCTA500TransformerEval()
        pass

    def __call__(self, image, target):
        # modal_list = [F.to_tensor(np.moveaxis(modal_list[0], 0, -1)), F.to_tensor(np.moveaxis(modal_list[1], 0, -1))]
        image = torch.from_numpy(image)
        # target = torch.as_tensor(np.array(target), dtype=torch.int64)
        target = torch.from_numpy(np.array(target, dtype='long'))
        return image, target

def get_nnunet_transformer(train, hypes):
    """
    增强灰度数据集的数据增强
    Args:
        hypes: 配置文件
        train: 是否是训练集

    Returns:

    """
    mean = (0.709, 0.381, 0.224)
    std = (0.127, 0.079, 0.043)
    base_size = 565
    crop_size = 480

    if train:
        return nnUnetPresetTrain(**hypes['augmentor']['args'])
    else:
        return nnUnetPresetEval(**hypes['augmentor']['args'])


def get_enhance_gray_transformer(train, hypes):
    """
    增强灰度数据集的数据增强
    Args:
        hypes: 配置文件
        train: 是否是训练集

    Returns:

    """
    mean = (0.709, 0.381, 0.224)
    std = (0.127, 0.079, 0.043)
    base_size = 565
    crop_size = 480

    if train:
        return EnhanceGrayPresetTrain(**hypes['augmentor']['args'])
    else:
        return EnhanceGrayPresetEval(**hypes['augmentor']['args'])


def get_octa500_transformer(train, hypes):
    """
    OCTA500的数据增强
    Args:
        hypes: 配置文件
        train: 是否是训练集

    Returns:

    """
    mean = (0.709, 0.381, 0.224)
    std = (0.127, 0.079, 0.043)
    base_size = 565
    crop_size = 480

    if train:
        return OCTA500PresetTrain(**hypes['augmentor']['args'])
    else:
        return OCTA500PresetEval(**hypes['augmentor']['args'])

def get_octa500_2d_transformer(train, hypes):
    """
    OCTA500的数据增强
    Args:
        hypes: 配置文件
        train: 是否是训练集

    Returns:

    """
    mean = (0.709, 0.381, 0.224)
    std = (0.127, 0.079, 0.043)
    base_size = 565
    crop_size = 480

    if train:
        return OCTA5002DPresetTrain(**hypes['augmentor']['args'])
    else:
        return OCTA5002DPresetEval(**hypes['augmentor']['args'])
