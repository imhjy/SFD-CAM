import numpy as np
import torch
from scipy.ndimage import rotate
from PIL import Image
import cv2
from torch import Tensor
from scipy.ndimage import binary_fill_holes
from skimage import measure

def inverse_polar_transform(image):
    """
       OpenCV极坐标逆变换函数
       参数:
           image
       返回:
           restored_img: 还原后的RGB图像（0-1 float32）
       """
    # 将图像转换为OpenCV所需格式
    device = None
    if isinstance(image, torch.Tensor):
        device = image.device
        image = image.cpu().numpy()
        if image.ndim == 3 and image.shape[0] == 1:
            image = np.squeeze(image, axis=0)

    img = rotate(image, 90, reshape=False)  # 双线性
    # polar_img_cv = (img * 255).astype(np.uint8)
    # 获取图像尺寸
    h, w = img.shape
    center = (w // 2, h // 2)  # 假设ROI已经是中心裁剪的视盘区域
    max_radius = min(center[0], center[1])  # 最大变换半径
    # 图像逆变换（使用线性插值）
    restored_img = cv2.warpPolar(
        src=img,
        dsize=[h, w],  # (width, height)
        center=center,
        maxRadius=max_radius,
        flags=cv2.WARP_FILL_OUTLIERS + cv2.WARP_INVERSE_MAP
    )
    # restored_img = cv2.cvtColor(restored_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    if device is not None:
        restored_img = torch.from_numpy(restored_img).to(device)
    return restored_img


def inverse_polar_transform_batch(image):
    """
       OpenCV极坐标逆变换函数(批量)
       参数:
           image
       返回:
           restored_img: 还原后的RGB图像（0-1 float32）
       """
    # 将图像转换为OpenCV所需格式
    is_tensor = isinstance(image, torch.Tensor)
    if is_tensor:
        device = image.device
        image_np = image.cpu().numpy()
    else:
        image_np = np.asarray(image)

    # 初始化输出数组
    batch_size = image_np.shape[0]
    restored_imgs = []

    for i in range(batch_size):
        img = image_np[i]  # 获取第i个样本 [H, W]
        img = rotate(img, 90, reshape=False)  # 旋转90度（双线性插值）

        # 极坐标逆变换
        h, w = img.shape
        center = (w // 2, h // 2)
        max_radius = min(center[0], center[1])
        img = np.array(img, dtype=np.uint8)
        restored_img = cv2.warpPolar(
            src=img,
            dsize=(w, h),  # OpenCV要求 (width, height)
            center=center,
            maxRadius=max_radius,
            flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR
        )
        restored_imgs.append(restored_img)

        # 合并Batch
    restored_imgs_np = np.stack(restored_imgs, axis=0)  # [B, H, W]

    # 返回与输入相同的类型
    if is_tensor:
        return torch.from_numpy(restored_imgs_np).to(device)
    else:
        return restored_imgs_np


def keep_maximum_connectivity(disc_map: np.ndarray, cup_map: np.ndarray):
    """
    后处理, 只保留最大连通域, 孔洞填充
    Args:
        disc_map: 预测的图片 ndarray((H,W))
        cup_map: 预测的图片 ndarray((H,W))
    Returns:

    """
    # def process(img):
    #     retval, labels, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)
    #
    #     # 找到面积最大的连通域（跳过背景 stats[0]）
    #     if retval <= 1:  # 只有背景（无连通域）
    #         return np.zeros_like(img)
    #     try:
    #         max_area_idx = np.argmax(stats[1:, 4]) + 1  # +1 因为跳过了背景
    #     except ValueError:
    #         print(f'未找到连通域.')
    #
    #     # 只保留最大连通域
    #     max_region = np.where(labels == max_area_idx, 128, 0).astype(np.uint8)
    #     return max_region
    def process(img):
        # 跳过全黑图像
        if not np.any(img):
            return img

        # 转换为二值图像（0和255）以便连通区域分析
        binary = np.zeros_like(img, dtype=np.uint8)
        binary[img != 0] = 255  # 所有前景设为255

        # 标记连通区域
        labeled = measure.label(binary)
        regions = measure.regionprops(labeled)

        if regions:
            # 找到面积最大的连通区域
            max_region_idx = np.argmax([r.area for r in regions]) + 1  # 标签从1开始

            # 只保留最大连通区域
            processed = np.zeros_like(binary)
            processed[labeled == max_region_idx] = 255

            # 填充孔洞
            processed = binary_fill_holes(processed)

            # 还原前景值为127
            result = np.zeros_like(img)
            result[processed != 0] = 128
            return result
        else:
            return img  # 没有连通区域时返回原图

    return process(disc_map), process(cup_map)


def check_cup_in_disc_area(disc_map: np.ndarray, cup_map: np.ndarray):
    """
    让视杯所有内容都在视盘内
    Parameters
    ----------
    disc_map: 背景为0, 前景为128
    cup_map: 背景为0, 前景为128

    Returns (disc_map, cup_map
    -------
    """
    # 转换为布尔型区域掩码[6,7](@ref)
    disc_mask = disc_map.astype(bool)  # 视盘有效区域
    cup_mask = cup_map.astype(bool)  # 视杯原始区域

    # 计算非法区域：视杯在视盘外的部分[8](@ref)
    overflow_mask = np.logical_and(cup_mask, ~disc_mask)

    # 生成修正后的视杯, 将非法区域置为0
    corrected_cup = np.where(overflow_mask, 0, cup_map)

    # 有效性验证（可选）
    if np.any(overflow_mask):
        print(f"发现{np.sum(overflow_mask)}个越界像素已清除")

    return disc_map, corrected_cup.astype(np.uint8)


def ellipse_fitting(disc_map: np.ndarray, cup_map: np.ndarray):
    """
    视杯视盘椭圆拟合核心算法
    实现原理：通过形态学优化轮廓，使用最小二乘法拟合椭圆参数
    Parameters
    ----------
    disc_map : 视盘二值图(背景0/前景128)
    cup_map  : 视杯二值图(背景0/前景128)
    """

    def process_mask(mask):
        # 二值化转换：128->255，0->0
        binary = np.where(mask == 128, 255, 0).astype(np.uint8)

        # 形态学闭运算闭合孔洞
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

        # 轮廓提取与筛选
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            print(f'轮廓未找到.')
            return mask

        # 选择最大面积轮廓
        max_contour = max(contours, key=cv2.contourArea)

        # 创建全黑背景的模板图像（与原始mask同尺寸）
        output = np.zeros_like(mask, dtype=np.uint8)

        # 椭圆拟合验证
        if len(max_contour) >= 5:
            ellipse = cv2.fitEllipse(max_contour)
            # 在模板图像上绘制填充椭圆（颜色128，厚度-1表示填充）
            cv2.ellipse(output, ellipse, color=128, thickness=-1)

            return output
        return mask

    # 分别处理视盘和视杯
    disc_ellipse = process_mask(disc_map)
    cup_ellipse = process_mask(cup_map)

    return disc_ellipse, cup_ellipse
