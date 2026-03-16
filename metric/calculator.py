import cv2
import torch
import numpy as np
from sklearn.metrics import roc_curve, auc

from utils.distributed_utils import DiceCoefficient
from utils.post_process import inverse_polar_transform


def calculate_absolute_cdr(predict, target, is_polar=False):
    def calculate_cdr(disc_map, cup_map):
        """计算杯盘比（Cup-to-Disc Ratio）

        参数：
            disc_map : Tensor - 视盘二值化图像矩阵
            cup_map : Tensor - 视杯二值化图像矩阵

        返回：
            CDR : float - 杯盘比
        """
        if isinstance(disc_map, torch.Tensor):
            disc_map = disc_map.cpu().numpy()
            if disc_map.ndim == 3 and disc_map.shape[0] == 1:
                disc_map = np.squeeze(disc_map, axis=0)

        if isinstance(cup_map, torch.Tensor):
            cup_map = cup_map.cpu().numpy()
            if cup_map.ndim == 3 and cup_map.shape[0] == 1:
                cup_map = np.squeeze(cup_map, axis=0)

        disc_rows = np.where(disc_map > 0)[0]
        cup_rows = np.where(cup_map > 0)[0]

        # 计算视盘直径
        disc_dia = (np.max(disc_rows) - np.min(disc_rows)) if disc_rows.size > 0 else 1

        # 计算视杯直径
        cup_dia = (np.max(cup_rows) - np.min(cup_rows)) if cup_rows.size > 0 else 1

        # 计算杯盘比（防止除零错误）
        CDR = cup_dia / disc_dia if disc_dia != 0 else 0.0

        return CDR

    predict = predict.clone()
    target = target.clone()
    """计算绝对杯盘比（Absolute Cup-to-Disc Ratio）"""
    disc_target = ((target == 1) | (target == 2)).long()
    cup_target = (target == 2).long()

    # disc_predict = torch.index_select(predict, dim=1, index=torch.tensor([0, 1], device=predict.device)).argmax(1).long()
    # cup_predict = torch.index_select(predict, dim=1, index=torch.tensor([0, 2], device=predict.device)).argmax(1).long()
    disc_predict = ((predict == 1) | (predict == 2)).long()
    cup_predict = (predict == 2).long()
    if is_polar:
        # 做一个极坐标转换
        disc_target = inverse_polar_transform(disc_target)
        cup_target = inverse_polar_transform(cup_target)
        disc_predict = inverse_polar_transform(disc_predict)
        cup_predict = inverse_polar_transform(cup_predict)

    gt_cdr = calculate_cdr(disc_target, cup_target)
    pre_cdt = calculate_cdr(disc_predict, cup_predict)
    return abs(gt_cdr - pre_cdt)


def calculate_absolute_cdar(predict, target, is_polar=False):
    # TODO """计算绝对杯盘面积比（Absolute Cup-to-Disc Area Ratio）"""
    def calculate_cdar(disc_map, cup_map):
        if isinstance(disc_map, torch.Tensor):
            disc_map = disc_map.cpu().numpy()
            if disc_map.ndim == 3 and disc_map.shape[0] == 1:
                disc_map = np.squeeze(disc_map, axis=0)

        if isinstance(cup_map, torch.Tensor):
            cup_map = cup_map.cpu().numpy()
            if cup_map.ndim == 3 and cup_map.shape[0] == 1:
                cup_map = np.squeeze(cup_map, axis=0)

        # 确保数据是uint8类型（OpenCV要求）
        disc_map = disc_map.astype(np.uint8)
        cup_map = cup_map.astype(np.uint8)

        # 计算视盘面积（最大连通区域）
        disc_num_labels, disc_labels, disc_stats, _ = cv2.connectedComponentsWithStats(disc_map)
        if disc_num_labels > 1:
            # 跳过背景（索引0），找出面积最大的连通区域
            disc_areas = disc_stats[1:, cv2.CC_STAT_AREA]  # 所有前景区域面积
            disc_largest_label = np.argmax(disc_areas) + 1  # +1 因为跳过了背景
            disc_area = disc_areas[disc_largest_label - 1]
        else:
            disc_area = 0  # 没有前景区域

        # 计算视杯面积（最大连通区域）
        cup_num_labels, cup_labels, cup_stats, _ = cv2.connectedComponentsWithStats(cup_map)
        if cup_num_labels > 1:
            cup_areas = cup_stats[1:, cv2.CC_STAT_AREA]
            cup_largest_label = np.argmax(cup_areas) + 1
            cup_area = cup_areas[cup_largest_label - 1]
        else:
            cup_area = 0

        # 计算面积比（避免除零错误）
        if disc_area > 0:
            cdr = cup_area / disc_area
        else:
            cdr = 0.0  # 视盘区域不存在时返回0

        return cdr

    predict = predict.clone()
    target = target.clone()
    disc_target = ((target == 1) | (target == 2)).long()
    cup_target = (target == 2).long()

    # disc_predict = torch.index_select(predict, dim=1, index=torch.tensor([0, 1], device=predict.device)).argmax(1).long()
    # cup_predict = torch.index_select(predict, dim=1, index=torch.tensor([0, 2], device=predict.device)).argmax(1).long()
    disc_predict = ((predict == 1) | (predict == 2)).long()
    cup_predict = (predict == 2).long()
    if is_polar:
        # 做一个极坐标转换
        disc_target = inverse_polar_transform(disc_target)
        cup_target = inverse_polar_transform(cup_target)
        disc_predict = inverse_polar_transform(disc_predict)
        cup_predict = inverse_polar_transform(cup_predict)

    gt_cdar = calculate_cdar(disc_target, cup_target)
    pre_cdar = calculate_cdar(disc_predict, cup_predict)
    return abs(gt_cdar - pre_cdar)


def calculate_absolute_cdpr(predict, target, is_polar=False):
    """计算绝对杯盘周长比（Absolute Cup-to-Disc Perimeter Ratio）"""

    def calculate_perimeter(mask):
        """计算二值掩码中最大连通区域的周长"""
        # 寻找所有轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return 0.0

        # 找到面积最大的轮廓（对应最大连通区域）
        largest_contour = max(contours, key=cv2.contourArea)

        # 计算轮廓周长
        perimeter = cv2.arcLength(largest_contour, closed=True)

        return perimeter

    def calculate_cdpr(disc_map, cup_map):
        if isinstance(disc_map, torch.Tensor):
            disc_map = disc_map.cpu().numpy()
            if disc_map.ndim == 3 and disc_map.shape[0] == 1:
                disc_map = np.squeeze(disc_map, axis=0)

        if isinstance(cup_map, torch.Tensor):
            cup_map = cup_map.cpu().numpy()
            if cup_map.ndim == 3 and cup_map.shape[0] == 1:
                cup_map = np.squeeze(cup_map, axis=0)

            # 确保数据是uint8类型
        disc_map = disc_map.astype(np.uint8)
        cup_map = cup_map.astype(np.uint8)

        # 计算视盘最大连通块的周长
        disc_perimeter = calculate_perimeter(disc_map)

        # 计算视杯最大连通块的周长
        cup_perimeter = calculate_perimeter(cup_map)

        # 计算周长比（避免除零错误）
        if disc_perimeter > 0:
            cdpr = cup_perimeter / disc_perimeter
        else:
            cdpr = 0.0

        return cdpr

    predict = predict.clone()
    target = target.clone()
    disc_target = ((target == 1) | (target == 2)).long()
    cup_target = (target == 2).long()

    # disc_predict = torch.index_select(predict, dim=1, index=torch.tensor([0, 1], device=predict.device)).argmax(1).long()
    # cup_predict = torch.index_select(predict, dim=1, index=torch.tensor([0, 2], device=predict.device)).argmax(1).long()
    disc_predict = ((predict == 1) | (predict == 2)).long()
    cup_predict = (predict == 2).long()
    if is_polar:
        # 做一个极坐标转换
        disc_target = inverse_polar_transform(disc_target)
        cup_target = inverse_polar_transform(cup_target)
        disc_predict = inverse_polar_transform(disc_predict)
        cup_predict = inverse_polar_transform(cup_predict)

    gt_cdpr = calculate_cdpr(disc_target, cup_target)
    pre_cdpr = calculate_cdpr(disc_predict, cup_predict)
    return abs(gt_cdpr - pre_cdpr)


class CDRCalculator(object):
    """
        求绝对CDR值, 水平杯盘比, 垂直杯盘比, 面积杯盘比
    """

    def __init__(self, num_classes=2, ignore_index=255, is_polar=False):
        super().__init__()
        self.num = 0  # 平均次数
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.metrics_name = ['absolute_cdr', 'absolute_cdar', 'absolute_cdpr']
        self.is_polar = is_polar
        self.total_dict = {}
        for item in self.metrics_name:
            self.total_dict[item] = 0.0

    def update(self, target, predict):
        self.num = self.num + 1
        # 垂直杯盘比
        self.total_dict['absolute_cdr'] = self.total_dict['absolute_cdr'] + calculate_absolute_cdr(predict, target,
                                                                                                   self.is_polar)
        # 杯盘面积比
        self.total_dict['absolute_cdar'] = self.total_dict['absolute_cdar'] + calculate_absolute_cdar(predict, target,
                                                                                                      self.is_polar)
        # 杯盘周长比
        self.total_dict['absolute_cdpr'] = self.total_dict['absolute_cdpr'] + calculate_absolute_cdpr(predict, target,
                                                                                                      self.is_polar)

    def compute(self):
        metric_dict = {}
        for item in self.metrics_name:
            metric_dict[item] = self.total_dict[item] / self.num

        if self.num == 0:
            return None
        else:
            return metric_dict

    def reduce_from_all_processes(self):
        """
        进程同步
        Returns:

        """
        if not torch.distributed.is_available():
            return
        if not torch.distributed.is_initialized():
            return
        torch.distributed.barrier()


class SingleMetricCalculator(object):
    """
    求平均Dice指数
    """

    def __init__(self, num_classes=2, ignore_index=255):
        super().__init__()
        self.num = 0  # 平均次数
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.device = 'cpu'
        self.dice = None
        self.metrics_name = ['mean_dice']
        self.total_dict = {}
        for item in self.metrics_name:
            self.total_dict[item] = 0.0

    def update(self, target, predict):
        self.num = self.num + 1
        n = self.num_classes
        mat = torch.zeros((n, n), dtype=torch.int64, device=target.device)
        if self.device is None:
            self.device = target.device
        with torch.no_grad():
            # 寻找GT中为目标的像素索引
            k = (target >= 0) & (target < n)  # 去除255掩码的影响
            # 统计像素真实类别a[k]被预测成类别b[k]的个数(这里的做法很巧妙)
            inds = n * target[k].to(torch.int64) + predict[k]
            mat += torch.bincount(inds, minlength=n ** 2).reshape(n, n)
            matrix = mat.float()
            TP = torch.diag(matrix)
            FP = matrix.sum(axis=0) - TP  # 每列的和减去TP即为FP
            FN = matrix.sum(axis=1) - TP  # 每行的和减去TP即为FN
            # Dice系数
            dice = 2 * TP / (2 * TP + FP + FN + 1e-7)
            # 求和
            self.total_dict['mean_dice'] = self.total_dict['mean_dice'] + dice
            # self.total_dict['absolute_cdr'] = self.total_dict['absolute_cdr'] + calculate_absolute_cdr(predict, target)

    def compute(self):
        metric_dict = {}
        for item in self.metrics_name:
            metric_dict[item] = 0

        if self.num == 0:
            return metric_dict
        else:
            for item in self.metrics_name:
                metric_dict[item] = self.total_dict[item] / self.num
            return metric_dict


class Calculator(object):
    """
    用于计算各种指标的组合类
    https://blog.csdn.net/wsljqian/article/details/99435808
    F1-score
    Dice-score
    accuracy
    Recall
    Precision
    Sensitivity 敏感度
    Specificity 特异度
    mAP
    https://blog.csdn.net/liujh845633242/article/details/102938143
    FROC
    ROC
    AUC
    先 update , 再 compute
    """

    def __init__(self, num_classes=5, ignore_index=255):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.mat = None
        self.roc_dict = None
        self.device = 'cpu'
        self.single_metric_calculator = SingleMetricCalculator(num_classes, ignore_index)

    def update(self, target, predict):
        """
        更新混淆矩阵
        Args:
            target:
            predict:

        Returns:

        """
        assert isinstance(target, torch.Tensor) and isinstance(predict, torch.Tensor), '输入数据必须是Tensor类型!'
        # target = target.flatten()
        # predict = predict.argmax(1).flatten()
        self.single_metric_calculator.update(target, predict)
        n = self.num_classes
        if self.mat is None:
            # 创建混淆矩阵
            self.mat = torch.zeros((n, n), dtype=torch.int64, device=target.device)
            self.device = target.device
        with torch.no_grad():
            # 寻找GT中为目标的像素索引
            k = (target >= 0) & (target < n)  # 去除255掩码的影响
            # 统计像素真实类别a[k]被预测成类别b[k]的个数(这里的做法很巧妙)
            inds = n * target[k].to(torch.int64) + predict[k]
            self.mat += torch.bincount(inds, minlength=n ** 2).reshape(n, n)

    def update_roc_auc(self, target, predict):
        self.device = target.device
        n = self.num_classes
        if self.roc_dict is None:
            self.roc_dict = {'fpr': {}, 'tpr': {}, 'thresholds': {}, 'roc_auc': [], 'predict': {}, 'target': {}}
        with torch.no_grad():
            k = (target >= 0) & (target < n)  # 去除255掩码的影响
            target = target[k]
            for i in range(self.num_classes):
                predict_t = predict[:, i][k]
                if f'{i}' not in self.roc_dict['target']:
                    self.roc_dict['target'][f'{i}'] = torch.tensor([], device=self.device)
                if f'{i}' not in self.roc_dict['predict']:
                    self.roc_dict['predict'][f'{i}'] = torch.tensor([], device=self.device)
                self.roc_dict['target'][f'{i}'] = torch.cat((self.roc_dict['target'][f'{i}'], target))
                self.roc_dict['predict'][f'{i}'] = torch.cat((self.roc_dict['predict'][f'{i}'], predict_t))

    def compute(self):
        """
            计算基于 self.mat 的所有指标（按类返回向量形式）
            返回 self.compute_dict 并包含若干汇总指标
        """
        if self.mat is None:
            raise ValueError("混淆矩阵为空，请先调用 update() 填充混淆矩阵。")

        device = self.device if hasattr(self, "device") else self.mat.device
        eps = 1e-7

        # 转为 float，避免整型除法问题
        matrix = self.mat.float().to(device)

        # 每类的 TP / FP / FN / TN
        TP = torch.diag(matrix)  # (num_classes,)
        FP = matrix.sum(dim=0) - TP  # 列和减去对角 -> 预测为该类但实际不是
        FN = matrix.sum(dim=1) - TP  # 行和减去对角 -> 实际为该类但预测不是
        total = matrix.sum()  # 总样本数（scalar）
        TN = total - (TP + FP + FN)  # 剩下为 TN

        # 保证都是 float tensor
        TP = TP.float()
        FP = FP.float()
        FN = FN.float()
        TN = TN.float()

        # 定义安全除法（当分母为0时返回0）
        def safe_div(num, den):
            return torch.where(den > 0, num / (den + eps), torch.zeros_like(num))

        # 全局/逐类指标
        accuracy_global = safe_div(TP.sum(), total)  # 全局准确率 (scalar)
        accuracy = safe_div(TP + TN, TP + TN + FP + FN)  # 每类准确率 (num_classes,)

        specificity = safe_div(TN, TN + FP)  # 特异性 (TN / (TN + FP))
        sensitivity = safe_div(TP, TP + FN)  # 敏感度/召回 (TP / (TP + FN))
        recall = sensitivity  # alias
        precision = safe_div(TP, TP + FP)  # 精确率 (TP / (TP + FP))

        dice = safe_div(2 * TP, 2 * TP + FP + FN)  # Dice
        iou = safe_div(TP, TP + FP + FN)  # IoU

        # F1 = 2 * precision * recall / (precision + recall)
        f1_denom = precision + recall
        f1_score = safe_div(2 * precision * recall, f1_denom)

        # 汇总指标（按需可删改）
        mean_dice = None
        # 尝试从 single_metric_calculator 获取 mean_dice（如果实现了）
        try:
            single_metric_dict = self.single_metric_calculator.compute()
            mean_dice = single_metric_dict.get('mean_dice', None)
        except Exception:
            mean_dice = float(dice.mean().item())

        mean_iou = float(iou.mean().item())
        mean_precision = float(precision.mean().item())
        mean_recall = float(recall.mean().item())

        # 保存 compute_dict
        self.compute_dict = {
            'TP': TP,
            'TN': TN,
            'FP': FP,
            'FN': FN,
            'support': (TP + FN),  # 每类真实样本数
            'accuracy_global': accuracy_global,
            'accuracy': accuracy,
            'specificity': specificity,
            'sensitivity': sensitivity,
            'recall': recall,
            'precision': precision,
            'dice': dice,
            'iou': iou,
            'f1_score': f1_score,
            'mean_precision': mean_precision,
            'mean_recall': mean_recall,
            'mean_iou': mean_iou,
            'mean_dice': mean_dice
        }

        return self.compute_dict

    def reduce_from_all_processes(self):
        """
        进程同步
        Returns:

        """
        if not torch.distributed.is_available():
            return
        if not torch.distributed.is_initialized():
            return
        torch.distributed.barrier()
        torch.distributed.all_reduce(self.mat)

    def __str__(self):
        self.compute_dict = self.compute()
        return (
            'Global Accuracy: {:.3f}\n'
            'Accuracy: {}\n'
            'Specificity: {}\n'
            'Sensitivity: {}\n'
            'Recall: {}\n'
            'Precision: {}\n'
            'Dice: {}\n'
            'Mean Dice: {}\n'
            'F1-Score: {}\n'
            'IoU: {}\n'
            'Mean IoU: {:.3f}'
        ).format(
            self.compute_dict['accuracy_global'].item() * 100,
            ['{:.1f}'.format(i) for i in (self.compute_dict['accuracy'] * 100).tolist()],
            ['{:.1f}'.format(i) for i in (self.compute_dict['specificity'] * 100).tolist()],
            ['{:.1f}'.format(i) for i in (self.compute_dict['sensitivity'] * 100).tolist()],
            ['{:.1f}'.format(i) for i in (self.compute_dict['recall'] * 100).tolist()],
            ['{:.1f}'.format(i) for i in (self.compute_dict['precision'] * 100).tolist()],
            ['{:.1f}'.format(i) for i in (self.compute_dict['dice'] * 100).tolist()],
            ['{:.1f}'.format(i) for i in (self.compute_dict['mean_dice'] * 100).tolist()],
            ['{:.1f}'.format(i) for i in (self.compute_dict['f1_score'] * 100).tolist()],
            ['{:.1f}'.format(i) for i in (self.compute_dict['iou'] * 100).tolist()],
            self.compute_dict['iou'].mean().item() * 100
        )

    @staticmethod
    def mean_calculator_list(calculator_list):
        """
        求多个calculator的n折交叉检验最后结果
        Args:
            calculator_list: calculator列表
        Returns:
            所有指标的Dict
        """
        assert len(calculator_list) > 0, 'mean_calculator_list方法入参列表长度必须大于0!'
        compute_num = len(calculator_list)
        keys = calculator_list[0].compute_dict.keys()
        result = {}
        for key in keys:
            value = None
            for calc in calculator_list:
                if value == None:
                    value = calc.compute_dict[key]
                else:
                    value = value + calc.compute_dict[key]
            result[key] = value / compute_num
        return result


def cal_Dice(img1, img2):
    """
    计算每个类别的 Dice 系数
    Args:
        img1: 预测图 (numpy数组)
        img2: 标签图 (numpy数组)
    Returns:
        dice: shape=(classnum, 1)，每类 Dice 值
    """
    classnum = int(img2.max())
    dice = np.zeros(classnum)
    for i in range(classnum):
        imga = img1 == i + 1
        imgb = img2 == i + 1
        intersection = np.sum(imga & imgb)
        denominator = np.sum(imga) + np.sum(imgb)
        if denominator == 0:
            dice[i] = 1.0  # 若该类不存在，视为完美匹配
        else:
            dice[i] = 2 * intersection / (denominator + 1e-5)
    return dice

class ConfusionMatrixMetric:
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.cm = np.zeros((num_classes, num_classes), dtype=np.int64)
        self.compute_dict = {}
        self.mean_dice = 0
        self.num_count = 0

    def _fast_cm(self, preds, targets):
        mask = (targets >= 0) & (targets < self.num_classes)
        hist = np.bincount(
            self.num_classes * targets[mask].astype(int) + preds[mask].astype(int),
            minlength=self.num_classes ** 2
        ).reshape(self.num_classes, self.num_classes)
        return hist

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        self.mean_dice = self.mean_dice + cal_Dice(preds.detach().cpu().numpy(), targets.detach().cpu().numpy())
        self.num_count += 1
        preds = preds.detach().cpu().numpy().flatten()
        targets = targets.detach().cpu().numpy().flatten()
        self.cm += self._fast_cm(preds, targets)


    def compute(self):
        cm = self.cm.astype(np.float64)
        TP = np.diag(cm)
        FP = cm.sum(axis=0) - TP
        FN = cm.sum(axis=1) - TP
        TN = cm.sum() - (TP + FP + FN)

        eps = 1e-7
        # metrics
        iou = TP / (TP + FP + FN + eps)
        dice = 2 * TP / (2 * TP + FP + FN + eps)
        precision = TP / (TP + FP + eps)
        recall = TP / (TP + FN + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)

        acc_global = TP.sum() / (cm.sum() + eps)
        acc_per_class = (TP + TN) / (TP + TN + FP + FN + eps)

        metrics = {
            "iou": iou,
            "dice": dice,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "accuracy": acc_per_class,
            "accuracy_global": acc_global,
            "mean_iou": np.mean(iou[1:]),
            "mean_dice": np.mean(dice[1:]),
            "mean_f1": np.mean(f1[1:]),
            "mean_accuracy": np.mean(acc_per_class[1:]),
            "sample_mean_dice": self.mean_dice / self.num_count,
        }

        self.compute_dict = metrics
        return self.compute_dict

    def reset(self):
        self.cm = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def __str__(self):
        self.compute()

        lines = []
        lines.append("=== ConfusionMatrixMetric Results ===")
        lines.append(f"Global Accuracy:    {self.compute_dict['accuracy_global']:.4f}")
        lines.append(f"Mean Accuracy:      {self.compute_dict['mean_accuracy']:.4f}")
        lines.append(f"Mean IoU:           {self.compute_dict['mean_iou']:.4f}")
        lines.append(f"Mean Dice:          {self.compute_dict['mean_dice']:.4f}")
        lines.append(f"Mean F1:            {self.compute_dict['mean_f1']:.4f}")
        lines.append(f"Mean Sample Dice:   {np.mean(self.compute_dict['sample_mean_dice']):.4f}")
        lines.append("Per-class metrics:")
        sample_mean_dice = np.insert(self.compute_dict['sample_mean_dice'], 0, 0)
        for c in range(self.num_classes):
            lines.append(
                f"  Class {c}: "
                f"Acc={self.compute_dict['accuracy'][c]:.4f}, "
                f"IoU={self.compute_dict['iou'][c]:.4f}, "
                f"Dice={self.compute_dict['dice'][c]:.4f}, "
                f"Precision={self.compute_dict['precision'][c]:.4f}, "
                f"Recall={self.compute_dict['recall'][c]:.4f}, "
                f"F1={self.compute_dict['f1_score'][c]:.4f}, "
                f"Mean Sample Dice={sample_mean_dice[c]:.4f}"
            )
        return "\n".join(lines)



class MetricsCalculator:
    def __init__(self):
        self.reset()

    def reset(self):
        """重置所有累计指标"""
        self.total_dice = 0.0
        self.total_miou = 0.0
        self.total_cavf_iou = None
        self.count = 0

    def _cal_dice(self, img1, img2):
        """计算 Dice 系数"""
        # 如果输入是 torch.Tensor，就转为 numpy
        if isinstance(img1, torch.Tensor):
            img1 = img1.detach().cpu().numpy()
        if isinstance(img2, torch.Tensor):
            img2 = img2.detach().cpu().numpy()
        shape = img1.shape
        I = np.sum((img1 >= 1) & (img2 >= 1))
        U = np.sum((img1 >= 1) | (img2 >= 1))
        return 2 * I / (I + U + 1e-5)

    def _cal_miou(self, img1, img2):
        """计算 mean IoU"""
        # 如果输入是 torch.Tensor，就转为 numpy
        if isinstance(img1, torch.Tensor):
            img1 = img1.detach().cpu().numpy()
        if isinstance(img2, torch.Tensor):
            img2 = img2.detach().cpu().numpy()
        classnum = int(img2.max())
        iou = np.zeros((classnum, 1))
        for i in range(classnum):
            imga = img1 == i + 1
            imgb = img2 == i + 1
            imgi = imga & imgb
            imgu = imga | imgb
            if np.sum(imgu) == 0:
                iou[i] = 1.0  # 避免除零，如果该类不存在则视为完美匹配
            else:
                iou[i] = np.sum(imgi) / np.sum(imgu)
        return np.mean(iou)

    def _cal_cavf_iou(self, img1, img2):
        """计算每类 IoU"""
        # 如果输入是 torch.Tensor，就转为 numpy
        if isinstance(img1, torch.Tensor):
            img1 = img1.detach().cpu().numpy()
        if isinstance(img2, torch.Tensor):
            img2 = img2.detach().cpu().numpy()
        classnum = int(img2.max())
        iou = np.zeros((classnum, 1))
        for i in range(classnum):
            imga = img1 == i + 1
            imgb = img2 == i + 1
            imgi = imga & imgb
            imgu = imga | imgb
            if np.sum(imgu) == 0:
                iou[i] = 1.0
            else:
                iou[i] = np.sum(imgi) / np.sum(imgu)
        return iou

    def update(self, pred, target):
        """更新一次计算结果"""
        dice = self._cal_dice(pred, target)
        miou = self._cal_miou(pred, target)
        cavf_iou = self._cal_cavf_iou(pred, target)

        self.total_dice += dice
        self.total_miou += miou

        if self.total_cavf_iou is None:
            self.total_cavf_iou = cavf_iou
        else:
            # 对每类 IoU 求平均（维度对齐）
            min_len = min(len(self.total_cavf_iou), len(cavf_iou))
            self.total_cavf_iou[:min_len] += cavf_iou[:min_len]

        self.count += 1

    def get_metrics(self):
        """返回当前平均指标"""
        if self.count == 0:
            return {
                'Dice': 0.0,
                'mIoU': 0.0,
                'cavfIoU': np.zeros((1, 1))
            }

        avg_cavf_iou = self.total_cavf_iou / self.count
        return {
            'Dice': self.total_dice / self.count,
            'mIoU': self.total_miou / self.count,
            'cavfIoU': avg_cavf_iou
        }

    def __str__(self):
        """格式化输出指标"""
        metrics = self.get_metrics()
        cavf_mean = np.mean(metrics['cavfIoU'])
        return (
            f"Dice: {metrics['Dice']:.4f} | "
            f"mIoU: {metrics['mIoU']:.4f} | "
            f"IoU: {metrics['cavfIoU']} | "
            f"Avg Class IoU: {cavf_mean:.4f}"
        )


class CollectIOU:
    def __init__(self):
        self.reset()

    def reset(self):
        """重置所有累计指标"""
        self.Capillary_IOU = []
        self.Artery_IOU = []
        self.Vein_IOU = []
        self.FAZ_IOU = []
        self.Mean_IOU = []

    def _cal_dice(self, img1, img2):
        """计算 Dice 系数"""
        # 如果输入是 torch.Tensor，就转为 numpy
        if isinstance(img1, torch.Tensor):
            img1 = img1.detach().cpu().numpy()
        if isinstance(img2, torch.Tensor):
            img2 = img2.detach().cpu().numpy()
        shape = img1.shape
        I = np.sum((img1 >= 1) & (img2 >= 1))
        U = np.sum((img1 >= 1) | (img2 >= 1))
        return 2 * I / (I + U + 1e-5)

    def _cal_miou(self, img1, img2):
        """计算 mean IoU"""
        # 如果输入是 torch.Tensor，就转为 numpy
        if isinstance(img1, torch.Tensor):
            img1 = img1.detach().cpu().numpy()
        if isinstance(img2, torch.Tensor):
            img2 = img2.detach().cpu().numpy()
        classnum = int(img2.max())
        iou = np.zeros((classnum, 1))
        for i in range(classnum):
            imga = img1 == i + 1
            imgb = img2 == i + 1
            imgi = imga & imgb
            imgu = imga | imgb
            if np.sum(imgu) == 0:
                iou[i] = 1.0  # 避免除零，如果该类不存在则视为完美匹配
            else:
                iou[i] = np.sum(imgi) / np.sum(imgu)
        return np.mean(iou)

    def _cal_cavf_iou(self, img1, img2):
        """计算每类 IoU"""
        # 如果输入是 torch.Tensor，就转为 numpy
        if isinstance(img1, torch.Tensor):
            img1 = img1.detach().cpu().numpy()
        if isinstance(img2, torch.Tensor):
            img2 = img2.detach().cpu().numpy()
        classnum = int(img2.max())
        iou = np.zeros((classnum, 1))
        for i in range(classnum):
            imga = img1 == i + 1
            imgb = img2 == i + 1
            imgi = imga & imgb
            imgu = imga | imgb
            if np.sum(imgu) == 0:
                iou[i] = 1.0
            else:
                iou[i] = np.sum(imgi) / np.sum(imgu)
        return iou

    def update(self, pred, target):
        """更新一次计算结果"""

        miou = self._cal_miou(pred, target)
        cavf_iou = self._cal_cavf_iou(pred, target)

        self.Mean_IOU.append(miou)
        self.Capillary_IOU.append(cavf_iou[0][0])
        self.Artery_IOU.append(cavf_iou[1][0])
        self.Vein_IOU.append(cavf_iou[2][0])
        self.FAZ_IOU.append(cavf_iou[3][0])

    def get_metrics(self):
        """返回当前平均指标"""
        return {
            "Mean_IOU": self.Mean_IOU,
            'Capillary_IOU': self.Capillary_IOU,
            'Artery_IOU': self.Artery_IOU,
            "Vein_IOU": self.Vein_IOU,
            "FAZ_IOU": self.FAZ_IOU,
        }


# ================= 示例使用 =================
if __name__ == "__main__":
    # 模拟两个简单的分割图
    img1 = np.array([[0, 1, 1], [2, 2, 0], [0, 1, 2]])
    img2 = np.array([[0, 1, 0], [2, 2, 0], [1, 1, 2]])

    metrics = MetricsCalculator()
    metrics.update(img1, img2)
    metrics.update(img1, img2)  # 可以多次更新

    print(metrics)  # 直接打印指标字符串
    print(metrics.get_metrics())  # 得到详细字典


# if __name__ == '__main__':
#     metric = ConfusionMatrixMetric(num_classes=5)  # 假设有4类
#
#     # 模拟批次
#     for _ in range(10):
#         preds = torch.randint(0, 5, ( 160, 160))  # 预测
#         target = torch.randint(0, 5, ( 160, 160))  # 标签
#         metric.update(preds, target)
#
#     # 计算指标
#     results = metric.compute()
#     print("Mean IoU:", results["mean_iou"])
#     print("Per-class Dice:", results["dice"])
#
#     print(str(metric))

if __name__ == "__main__":
    # 模拟两个简单的分割图
    img1 = np.array([[0, 1, 1], [2, 2, 0], [0, 1, 2]])
    img2 = np.array([[0, 1, 0], [2, 2, 0], [1, 1, 2]])

    metrics = MetricsCalculator()
    metrics.update(img1, img2)
    metrics.update(img1, img2)  # 可以多次更新

    print(metrics)  # 直接打印指标字符串
    print(metrics.get_metrics())  # 得到详细字典