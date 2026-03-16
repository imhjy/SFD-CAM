import os
import platform
import random

import cv2
import numpy as np
import torch
from torch import Tensor

from metric.calculator import Calculator, CDRCalculator
from utils.post_process import inverse_polar_transform, inverse_polar_transform_batch


def str_dict_from_tensor(obj):
    """
    将一个内容有Tensor的dict转换为普通形式
    Args:
        dict:

    Returns:

    """
    if isinstance(obj, dict):
        return {k: str_dict_from_tensor(v) for k, v in obj.items()}
    elif isinstance(obj, torch.Tensor):
        return obj.item() if obj.numel() == 1 else obj.tolist()
    elif isinstance(obj, list):
        return [str_dict_from_tensor(item) for item in obj]
    else:
        return obj


def replace_system_separator(path: str):
    sys = platform.system()
    if sys == "Linux":
        return path.replace('\\', '/')
    elif sys == "Windows":
        return path.replace('/', '\\')
    return path


def remove_small_areas_based_threshold(img: Tensor, threshold=5):
    """
    后处理, 去掉连通域小于阈值的预测前景区域
    Args:
        img: 预测的图片 Tensor((H,W))
        threshold: 阈值, 默认5, 根据不同数据集进行调整

    Returns:

    """
    device = img.device
    img = img.cpu().numpy().astype(np.uint8)
    # 使用cv2.connectedComponentsWithStats函数统计联通域信息
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)
    # 遍历stats数组，将小于阈值的联通域在labels中标记为0
    for i in range(1, stats.shape[0]):
        area = stats[i, 4]  # stats数组中第5列是面积信息
        if area < threshold:
            labels[labels == i] = 0

    # 将labels标记图二值化处理, 大于0的换成255
    labels = labels.astype(np.uint8)
    ret, labels = cv2.threshold(labels, 0, 255, cv2.THRESH_BINARY)
    labels[labels > 0] = 1
    return torch.tensor(labels, device=device)


def setup_seed(seed=3407):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True


def calculate_grad_norm(model):
    """
    计算当前模型的梯度, 使用前要把梯度裁剪的代码注释, 不然无效

    # 梯度裁剪（按范数裁剪）, 只要开始两个batch会到6, 后面全是1~5之间, 所以选择5
    # (如果使用模型初始化开始就是10多, 测试后发现加上梯度裁剪对模型收敛有坏处, 注释)
    # 参数更新量 ≈ 学习率 × 梯度范数
    # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
    Args:
        model: nn.Module

    Returns:

    """
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5


def get_loss(criterion, output, target, loss_weight=torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0]).cuda(), num_classes=5, ignore_idx=255):

    # disc_target = ((target == 1) | (target == 2)).long()
    # cup_target = (target == 2).long()

    # disc_output = torch.index_select(output, dim=1, index=torch.tensor([0, 1], device=output.device))
    # cup_output = torch.index_select(output, dim=1, index=torch.tensor([0, 2], device=output.device))
    #
    # # 分别是视盘和视杯, 其中视盘包含视杯
    # loss = criterion(disc_output, disc_target,
    #                  loss_weight[[0, 1]], num_classes=2, ignore_index=ignore_idx) \
    #        + criterion(cup_output, cup_target,
    #                    loss_weight[[0, 2]], num_classes=2, ignore_index=ignore_idx)


    # 分别是视盘和视杯, 其中视盘包含视杯
    loss = criterion(output, target, weight=loss_weight, num_classes=num_classes, ignore_index=ignore_idx)
    return loss


def get_disc_cup_calculator(hypes):
    """
    获取指标计算类
    """
    disc_calculator = Calculator(num_classes=2)
    cup_calculator = Calculator(num_classes=2)
    cdr_calculator = CDRCalculator(is_polar='Polar' in hypes['augmentor']['core_method'])
    return disc_calculator, cup_calculator, cdr_calculator


def update_disc_cup_calculator(disc_calculator: Calculator, cup_calculator: Calculator, cdr_calculator: CDRCalculator,
                               output, target, is_polar=False):
    """
    获取指标计算类
    output: 模型输出的原始数据
    target: 标签
    """
    if is_polar:
        target = inverse_polar_transform(target)
    disc_target = ((target == 1) | (target == 2)).long()
    cup_target = (target == 2).long()

    output_argmax = output.argmax(1)
    disc_output = (output_argmax >= 1).long()
    cup_output = (output_argmax == 2).long()
    # disc_output = torch.index_select(output, dim=1, index=torch.tensor([0, 1], device=output.device)).argmax(1)
    # cup_output = torch.index_select(output, dim=1, index=torch.tensor([0, 2], device=output.device)).argmax(1)
    if is_polar:
        disc_output = inverse_polar_transform(disc_output)
        cup_output = inverse_polar_transform(cup_output)

    final_mask = torch.zeros_like(disc_output).to(disc_output.device)
    final_mask[disc_output == 1] = 1  # 视盘=1
    final_mask[cup_output == 1] = 2  # 视杯=2

    cdr_calculator.update(target, final_mask)
    disc_calculator.update(disc_target.flatten(), disc_output.flatten())
    cup_calculator.update(cup_target.flatten(), cup_output.flatten())


def update_disc_cup_calculator_post(disc_calculator: Calculator, cup_calculator: Calculator,
                                    cdr_calculator: CDRCalculator, output, target, is_polar=False):
    """
    获取指标计算类
    """
    if isinstance(output, np.ndarray):
        output = torch.tensor(output).to(target.device)
    # 处理标签
    if is_polar:
        target = inverse_polar_transform(target)
    disc_target = ((target == 1) | (target == 2)).long()
    cup_target = (target == 2).long()

    disc_output = torch.logical_or(output == 128, output == 255).long()
    cup_output = (output == 255).long()

    final_mask = torch.zeros_like(disc_output).to(disc_output.device)
    final_mask[disc_output == 1] = 1  # 视盘=1
    final_mask[cup_output == 1] = 2  # 视杯=2

    cdr_calculator.update(target, final_mask)
    disc_calculator.update(disc_target.flatten(), disc_output.flatten())
    cup_calculator.update(cup_target.flatten(), cup_output.flatten())


class Predictor3Dto2DWrapper:
    def __init__(self, model: torch.nn.Module):
        self.model = model

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, D, H, W)
        return: (B, C_out, D, H, W)
        """
        # 送入2D模型，得到 (B, C_out, H, W)
        out_2d = self.model(x)
        out = out_2d['out']

        # 在 depth 维拼接回 (B, C_out, D, H, W)
        return out.unsqueeze(2)

class LogWriter(object):
    def __init__(self, *files):
        self.files = files  # 可以传多个目标，比如 sys.__stdout__ 和 log 文件

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()  # 确保实时写入

    def flush(self):
        for f in self.files:
            f.flush()

if __name__ == '__main__':
    tensor_dict = {
        'key1': torch.tensor(1),
        'key2': {
            'key3': torch.tensor(2),
            'key4': torch.tensor([3, 4, 5])
        },
        'key5': [torch.tensor(6), torch.tensor([7, 8])]
    }
    regular_dict = str_dict_from_tensor(tensor_dict)
    print(regular_dict)
