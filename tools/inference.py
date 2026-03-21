"""
推理验证
"""
import csv
import json
import os
import sys
import argparse
from typing import List, Optional

import numpy as np
import torch
from PIL import Image
from monai.inferers import sliding_window_inference

from data_utils.datasets import build_dataset
from metric.calculator import Calculator, ConfusionMatrixMetric, CollectIOU
from utils import train_utils
import utils.distributed_utils as utils
from utils.common import str_dict_from_tensor, remove_small_areas_based_threshold, replace_system_separator, \
    get_disc_cup_calculator, update_disc_cup_calculator, update_disc_cup_calculator_post, Predictor3Dto2DWrapper
import matplotlib.pyplot as plt
import pandas as pd

from utils.post_process import inverse_polar_transform, check_cup_in_disc_area, ellipse_fitting, \
    keep_maximum_connectivity


class_names = ['Background', 'Capillary', 'Artery', 'Vein', 'FAZ']


def parse_args():
    parser = argparse.ArgumentParser(description="推理参数")
    parser.add_argument('--model_dir', type=str,
                        default='../logs/compare/6mm/UNet',
                        help='模型路径')
    parser.add_argument('--save_vis', type=bool, default=True,
                        help='保存语义分割后的图像')
    parser.add_argument('--eval_epoch', type=int, default=None,
                        help='加载哪个epoch的模型, 为None加载最好的模型')
    opt = parser.parse_args()
    return opt


def _ensure_dir(path: str):
    folder = os.path.dirname(path)
    if folder and not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)


def _format_float(x, fmt="{:.6f}"):
    try:
        return fmt.format(float(x))
    except Exception:
        return "NA"


def write_csv(saved_path: str,
              calculator,  # ConfusionMatrixMetric instance
              info: str,
              fold: Optional[int] = 0,
              class_names: Optional[List[str]] = None,
              csv_name: str = "validate_results.csv"):
    """
    将单个 ConfusionMatrixMetric 写入 CSV（一行）。
    """
    if not hasattr(calculator, "compute_dict") or not calculator.compute_dict:
        calculator.compute()

    num_classes = calculator.num_classes
    if class_names is None:
        class_names = [f"class_{i}" for i in range(num_classes)]
    assert len(class_names) == num_classes

    # 生成字段名： info, fold, then per-class metrics, then summary metrics
    per_metric_keys = ["iou", "dice", "f1_score", "precision", "recall", "accuracy", "sample_dice"]
    fieldnames = ["info", "fold"]
    for cname in class_names:
        for k in per_metric_keys:
            fieldnames.append(f"{cname}_{k}")
    # 汇总/总体字段
    summary_keys = ["mean_iou", "mean_dice", "mean_sample_dice", "mean_f1", "mean_accuracy", "accuracy_global"]
    fieldnames.extend(summary_keys)

    csv_path = os.path.join(saved_path, csv_name)
    _ensure_dir(csv_path)
    write_header = not os.path.exists(csv_path)

    # 构建 row
    cd = calculator.compute_dict
    row = {"info": info, "fold": str(fold)}
    # per class
    for i, cname in enumerate(class_names):
        row[f"{cname}_iou"] = _format_float(cd["iou"][i])
        row[f"{cname}_dice"] = _format_float(cd["dice"][i])
        row[f"{cname}_f1_score"] = _format_float(cd["f1_score"][i])
        row[f"{cname}_precision"] = _format_float(cd["precision"][i])
        row[f"{cname}_recall"] = _format_float(cd["recall"][i])
        row[f"{cname}_accuracy"] = _format_float(cd["accuracy"][i])
        row[f"{cname}_sample_dice"] = _format_float(0) if i == 0 else _format_float(cd["sample_mean_dice"][i - 1])

    # summary
    # 注意：你的 compute_dict 里有 mean_* 字段（如果 compute() 做了 iou[1:] 的平均，这里直接取）
    row["mean_iou"] = _format_float(cd.get("mean_iou", np.mean(cd["iou"])))
    row["mean_dice"] = _format_float(cd.get("mean_dice", np.mean(cd["dice"])))
    row["mean_sample_dice"] = _format_float(cd.get("mean_sample_dice", np.mean(cd["sample_mean_dice"])))
    row["mean_f1"] = _format_float(cd.get("mean_f1", np.mean(cd["f1_score"])))
    row["mean_accuracy"] = _format_float(cd.get("mean_accuracy", np.mean(cd["accuracy"])))
    row["accuracy_global"] = _format_float(cd.get("accuracy_global", np.mean(cd["accuracy"])))

    # 写入 CSV（追加）
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        # 为保证列顺序一致，先构造 ordered_row
        ordered_row = {fn: row.get(fn, "NA") for fn in fieldnames}
        writer.writerow(ordered_row)


def write_summary_csv(saved_path: str,
                      calculator_list: List,
                      info: str,
                      class_names: Optional[List[str]] = None,
                      csv_name: str = "validate_results.csv"):
    assert len(calculator_list) > 0
    num_classes = calculator_list[0].num_classes
    if class_names is None:
        class_names = [f"class_{i}" for i in range(num_classes)]
    assert len(class_names) == num_classes

    per_metric_keys = ["iou", "dice", "f1_score", "precision", "recall", "accuracy", "sample_dice"]
    fieldnames = ["info", "fold"]
    for cname in class_names:
        for k in per_metric_keys:
            fieldnames.append(f"{cname}_{k}")
    summary_keys = ["mean_iou", "mean_dice", "mean_sample_dice", "mean_f1", "mean_accuracy", "accuracy_global"]
    fieldnames.extend(summary_keys)

    csv_path = os.path.join(saved_path, csv_name)
    _ensure_dir(csv_path)
    write_header = not os.path.exists(csv_path)

    # 确保所有计算器都计算一次并收集数据
    rows = []
    for i, calc in enumerate(calculator_list):
        if not hasattr(calc, "compute_dict") or not calc.compute_dict:
            calc.compute()
        cd = calc.compute_dict
        row = {"info": info, "fold": str(i + 1)}
        for j, cname in enumerate(class_names):
            row[f"{cname}_iou"] = float(np.asarray(cd["iou"][j]))
            row[f"{cname}_dice"] = float(np.asarray(cd["dice"][j]))
            row[f"{cname}_f1_score"] = float(np.asarray(cd["f1_score"][j]))
            row[f"{cname}_precision"] = float(np.asarray(cd["precision"][j]))
            row[f"{cname}_recall"] = float(np.asarray(cd["recall"][j]))
            row[f"{cname}_accuracy"] = float(np.asarray(cd["accuracy"][j]))
            row[f"{cname}_sample_dice"] = float(np.asarray(0)) if j == 0 else float(np.asarray(cd["sample_mean_dice"][j - 1]))
        row["mean_iou"] = float(np.asarray(cd.get("mean_iou", np.mean(cd["iou"]))))
        row["mean_dice"] = float(np.asarray(cd.get("mean_dice", np.mean(cd["dice"]))))
        row["mean_sample_dice"] = float(np.asarray(cd.get("mean_sample_dice", np.mean(cd["sample_mean_dice"]))))
        row["mean_f1"] = float(np.asarray(cd.get("mean_f1", np.mean(cd["f1_score"]))))
        row["mean_accuracy"] = float(np.asarray(cd.get("mean_accuracy", np.mean(cd["accuracy"]))))
        row["accuracy_global"] = float(np.asarray(cd.get("accuracy_global", np.mean(cd["accuracy"]))))
        rows.append(row)

    # 计算每个字段的 fold 平均值（排除 info/fold）
    agg = {}
    numeric_fields = [fn for fn in fieldnames if fn not in ("info", "fold")]
    for fn in numeric_fields:
        vals = []
        for r in rows:
            v = r.get(fn, None)
            if v is None:
                continue
            try:
                vals.append(float(v))
            except Exception:
                pass
        agg[fn] = float(np.mean(vals)) if len(vals) > 0 else float("nan")

    # 将单折写入 CSV 并最后写 mean 行
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for r in rows:
            # 将值格式化为字符串保留 6 位小数
            out = {}
            for fn in fieldnames:
                if fn in ("info", "fold"):
                    out[fn] = r.get(fn, "")
                else:
                    out[fn] = _format_float(r.get(fn, "NA"))
            writer.writerow(out)

        # 写 mean 行
        mean_row = {"info": info, "fold": "mean"}
        for fn in numeric_fields:
            mean_row[fn] = _format_float(agg[fn])
        # 写入
        ordered_mean = {fn: mean_row.get(fn, "NA") for fn in fieldnames}
        writer.writerow(ordered_mean)

    # 打印最终 summary 到控制台（可选）
    print("Summary written to:", csv_path)
    print(json.dumps({k: _format_float(v) for k, v in agg.items()}, indent=2))


def save_outputs_and_labels(outputs, labels, fold_idx, save_dir):
    from pathlib import Path
    """
    保存模型输出和标签到单个.pt文件中

    参数:
    outputs: 模型输出Tensor
    labels: 对应标签Tensor
    fold_idx: 文件编号（用于文件名）
    save_dir: 保存目录
    """
    # 确保保存目录存在
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # 构建文件路径
    file_path = Path(save_dir) / f"temp_fold{fold_idx}.pt"

    # 转换为浮点张量（确保数据类型一致）
    outputs = torch.cat([t.unsqueeze(0) for t in outputs], dim=0).detach().cpu()
    labels = torch.cat([t.unsqueeze(0) for t in labels], dim=0).detach().cpu()

    # 保存到文件
    torch.save({
        'outputs': outputs,
        'labels': labels,
    }, file_path)


def load_outputs_and_labels(fold_idx, save_dir, device="cpu"):
    from pathlib import Path
    """
    从.pt文件加载模型输出和标签

    参数:
    fold_idx: 文件编号
    save_dir: 保存目录
    device: 加载到哪个设备
    """
    file_path = Path(save_dir) / f"temp_fold{fold_idx}.pt"

    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")

    # 加载数据
    data = torch.load(file_path, map_location=device)

    # 返回输出和标签
    outputs = data['outputs']
    labels = data['labels']
    outputs_arr = []
    labels_arr = []
    for i in range(outputs.size()[0]):
        outputs_arr.append(outputs[i])
        labels_arr.append(labels[i])
    return outputs_arr, labels_arr


def save_image(image, save_path, file_name):
    """
    保存OCTA图像
    ----------
    disc_map: 视盘图
    cup_map 视杯图
    save_path: 保存位置
    file_name: 保存名称
    """
    color_map = {
        0: [0, 0, 0],  # 黑色
        1: [255, 255, 255],  # 白色
        2: [255, 0, 0],  # 红色
        3: [0, 255, 0],  # 绿色
        4: [255, 255, 0]  # 黄色
    }

    # 转换为numpy数组操作
    image_array = image.cpu().numpy()
    result_color = np.zeros((image.shape[1], image.shape[2], 3), dtype=np.float32)
    # 核心逻辑：视杯区域覆盖为255，其他区域继承视盘值
    # 创建彩色结果图像
    for class_idx, color in color_map.items():
        mask = (image_array == class_idx)
        for c in range(3):  # RGB三个通道
            result_color[:, :, c] = np.where(mask, color[c], result_color[:, :, c])

    # 生成图像并保存
    mask = Image.fromarray(result_color.astype(np.uint8))
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    mask.save(f"{os.path.join(save_path, file_name)}")
    return result_color


def remove_all_csv(base_path, fold_num_list):
    """
    删除所有验证的csv文件
    """
    for fold in fold_num_list:
        csv_path = os.path.join(base_path, f'fold-{fold}', 'validate_results.csv')
        if os.path.exists(csv_path):
            os.remove(csv_path)
    csv_path = os.path.join(base_path, 'validate_results.csv')
    if os.path.exists(csv_path):
        os.remove(csv_path)


def main(args, hypes):
    device = torch.device(hypes['device'])
    # 分割的分类数目 nun_classes + background
    num_classes = hypes['num-classes'] + 1
    # hypes['model']['args']['num_classes'] = num_classes
    # 第几折进行训练
    fold_num_list = hypes['train_params']['train_fold_list']
    calculator_list = []
    remove_all_csv(args.model_dir if args.model_dir else args.hypes_yaml, fold_num_list)
    global_calculator = ConfusionMatrixMetric(num_classes=num_classes)
    collectIOU = CollectIOU()
    for fold in fold_num_list:
        print('-----------------Dataset Building------------------')
        val_dataset = build_dataset(hypes, train=False, fold=fold)
        num_workers = hypes['num_workers']
        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=1,
                                                 num_workers=num_workers,
                                                 shuffle=False,
                                                 pin_memory=True,
                                                 collate_fn=val_dataset.collate_fn if hypes[
                                                                                          'input_type'] == '3d' else val_dataset.collate_fn_2d)
        print('---------------Creating Model------------------')
        model = train_utils.create_model(hypes)
        print('-----------------Load Pretrained Model------------------')
        load_path = os.path.join(args.model_dir, f'fold-{fold}')
        _, model, _, _, _, _ = train_utils.load_saved_model(load_path, model, None, None, None)
        model.to(device)
        print('-----------------Eval Step------------------')
        model.eval()
        calculator = ConfusionMatrixMetric(num_classes=num_classes)
        metric_logger = utils.MetricLogger(delimiter="  ")
        header = f'Test Fold [{fold}/{len(fold_num_list)}]:'

        output_arr = []
        label_arr = []
        with torch.no_grad():
            idx = 0
            for images, target in metric_logger.log_every(val_loader, 100, header):
                if hypes['input_type'] == '3d':
                    # 合并一下数据
                    image, target = images.to(device), target.to(device, dtype=torch.long)
                    # image = torch.stack([modal_list[0], modal_list[1]], dim=1)
                    # image = modal_list[1].unsqueeze(1)

                    predictor = Predictor3Dto2DWrapper(model)
                    output = sliding_window_inference(inputs=image, roi_size=(
                        hypes['dataset']['block_size'][0], hypes['dataset']['block_size'][1],
                        hypes['dataset']['block_size'][2]),
                                                      sw_batch_size=1,
                                                      predictor=predictor, overlap=0.25)  # 使用滑动窗口进行推理
                    output = output.squeeze(2)
                elif hypes['input_type'] == '2d':
                    image, target = images.unsqueeze(1).to(device), target.to(device,
                                                                              dtype=torch.long)

                    output = sliding_window_inference(inputs=image, roi_size=(
                        hypes['dataset']['block_size'][1],
                        hypes['dataset']['block_size'][2]),
                                                      sw_batch_size=1,
                                                      predictor=model, overlap=0.25)  # 使用滑动窗口进行推理
                    output = output['out']

                output_arr.append(output.to("cpu"))
                label_arr.append(target.to("cpu"))
                # ---- 保存推理图像 ----
                file_name = os.path.basename(val_dataset.label[idx])
                save_image(output.argmax(1),
                           os.path.join(load_path, 'save_images'),
                           os.path.splitext(file_name)[0] + '.bmp')

                calculator.update(output.argmax(1), target)
                collectIOU.update(output.argmax(1), target)
                # global_calculator.update(output.argmax(1), target)
                idx += 1

            # dice.reduce_from_all_processes()
        # draw_and_save_roc(calculator.roc_dict, load_path)
        val_info = str(calculator)
        print(f'CAVF info: \n {val_info}')
        # 将本次验证的结果写入文件
        write_csv(load_path, calculator, info='origin', class_names=class_names)
        calculator_list.append(calculator)
        save_outputs_and_labels(output_arr, label_arr, fold, args.model_dir)
        del model
    # 计算总体结果, 并且写入文件
    for fold in fold_num_list:
        output_arr, label_arr = load_outputs_and_labels(fold, args.model_dir)

        for i in range(len(output_arr)):
            output = output_arr[i]
            target = label_arr[i]
            global_calculator.update(output.argmax(1), target)
    write_summary_csv(args.model_dir, calculator_list, info='origin', class_names=class_names)

    write_csv(args.model_dir, global_calculator, info='global_origin', class_names=class_names)

    # 把data保存args.model_dir为一个json
    with open(os.path.join(args.model_dir, 'IOU.json'), 'w') as f:
        data = collectIOU.get_metrics()
        json.dump(data, f)

    # 删除所有temp_fold*.pt
    for file in os.listdir(args.model_dir):
        if file.startswith('temp_fold') and file.endswith('.pt'):
            os.remove(os.path.join(args.model_dir, file))



if __name__ == '__main__':

    model_dir_path = [
        '../logs/compare/3mm/SFDFormer',
    ]
    for path in model_dir_path:
        print('-----------------Analyze Config File------------------')
        args = parse_args()
        args.model_dir = path
        hypes = load_yaml(None, args)
        # modify_config(hypes)
        print(f'当前训练模型路径: {os.path.abspath(path)}')
        main(args, hypes)
