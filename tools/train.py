import csv
import gc
import inspect
import os
import sys

import numpy as np
from torch import nn

from utils.common import setup_seed,  get_loss, Predictor3Dto2DWrapper, LogWriter
from utils.train_utils import initialize_layers

root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-3])
sys.path.append(root_path)
sys.path.append('/projects/OCTA-Seg')
sys.path.append('/projects/OCTA-Seg/tools')
print(sys.executable)
import time
import datetime
import argparse
import torch
from monai.inferers import sliding_window_inference
from torch.utils.tensorboard import SummaryWriter

from data_utils.datasets import build_dataset
from hypes_yaml import yaml_utils
from loss import build_loss
from metric.calculator import  ConfusionMatrixMetric, MetricsCalculator
from utils import train_utils
from lr_schedular import build_lr_schedular
import utils.distributed_utils as utils
from utils.early_stopping import EarlyStopping
import torch.nn.functional as F
from inference import main as inference_main

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True # 自动寻找最优的卷积算法

def parse_args():
    parser = argparse.ArgumentParser(description="训练参数")
    parser.add_argument("--hypes_yaml", type=str,
                        default="../hypes_yaml/OCTA-500/IPN.yaml",
                        # default="../hypes_yaml/OCTA-500/CENet.yaml",
                        help='配置文件路径')
    parser.add_argument('--model_dir', type=str,
                        # default='../logs/model/SFDFormer2',
                        help='训练路径,与hypes_yaml二选一')
    parser.add_argument('--save_vis', type=bool, default=True,
                        help='保存语义分割后的图像')
    parser.add_argument('--eval_epoch', type=int, default=None,
                        help='加载哪个epoch的模型, 为None加载最好的模型')
    parser.add_argument('--immediately_inference', type=bool, default=False,
                        help='是否训练好后立即推理')
    args = parser.parse_args()

    return args


def main(args, hypes):
    global early_stopping
    device = torch.device(hypes['device'])

    print(f"模型名称: {hypes['name']}")
    print(f"设备: {device}")
    print(f"数据增强方法: {hypes['augmentor']['core_method']}")
    print(f"数据集: {hypes['dataset']['method']}")
    print(f"epoch数: {hypes['train_params']['epoches']}")

    batch_size = hypes['train_params']['batch_size']
    # 分割的分类数目 nun_classes + background
    num_classes = hypes['num-classes'] + 1
    # hypes['model']['args']['num_classes'] = num_classes
    # 第几折进行训练
    fold_num_list = hypes['train_params']['train_fold_list']
    # epoch数
    epochs = hypes['train_params']['epoches']
    f1_value = 0

    # 是否使用梯度裁剪
    is_use_grad_norm = 'grad_norm' in hypes and hypes['grad_norm']['use'] is True

    # 随机数种子
    if 'seed' in hypes and hypes['seed'] != -1:
        setup_seed(hypes['seed'])
    origin_path = None  # 防止创建多个文件夹
    for fold in fold_num_list:
        print(f"模型名称: {hypes['name']}")
        print('-----------------Dataset Building------------------')
        train_dataset = build_dataset(hypes, train=True, fold=fold)
        val_dataset = build_dataset(hypes, train=False, fold=fold)
        num_workers = hypes['num_workers']
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size,
                                                   num_workers=num_workers,
                                                   shuffle=True,
                                                   pin_memory=True,
                                                   drop_last=True,
                                                   collate_fn=train_dataset.collate_fn if hypes[
                                                                                              'input_type'] == '3d' else train_dataset.collate_fn_2d)

        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=1,
                                                 num_workers=num_workers,
                                                 pin_memory=True,
                                                 collate_fn=val_dataset.collate_fn if hypes[
                                                                                          'input_type'] == '3d' else val_dataset.collate_fn_2d)

        print('---------------Creating Model------------------')
        model = train_utils.create_model(hypes)
        # model.apply(initialize_layers)  # 进行模型初始化
        # 优化器
        # params_to_optimize = [p for p in model.parameters() if p.requires_grad]
        # optimizer = torch.optim.SGD(
        #     params_to_optimize,
        #     lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
        # )
        optimizer = train_utils.setup_optimizer(hypes, model)

        # 混合精度
        scaler = torch.amp.GradScaler(hypes['device']) if hypes['amp'] else None

        # 创建学习率更新策略，这里是每个step更新一次(不是每个epoch)
        num_steps = len(train_loader)
        # lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs, warmup=True)

        lr_scheduler = build_lr_schedular(optimizer, hypes, num_step=num_steps, epochs=epochs,
                                          **hypes['lr_scheduler']['args'])

        if num_classes == 2:
            # 设置cross_entropy中背景和前景的loss权重(根据自己的数据集进行设置)
            loss_weight = torch.as_tensor([1.0, 2.0], device=device)
        elif num_classes == 3:
            # 设置cross_entropy中背景和前景的loss权重(根据自己的数据集进行设置)
            loss_weight = torch.as_tensor([0.0, 1.0, 2.0], device=device)
        elif num_classes == 4:
            # 设置cross_entropy中背景和前景的loss权重(根据自己的数据集进行设置)
            loss_weight = torch.as_tensor([1.0, 1.0, 1.0, 1.0], device=device)
        elif num_classes == 5:
            # 设置cross_entropy中背景和前景的loss权重(根据自己的数据集进行设置)
            loss_weight = torch.as_tensor([1.0, 2.0, 2.0, 2.0, 2.0], device=device)
        else:
            loss_weight = None

        # 损失函数
        criterion = build_loss(hypes)
        if inspect.isclass(criterion):
            criterion = criterion(loss_weight=loss_weight)
        # criterion = nn.CrossEntropyLoss()

        lowest_val_epoch = -1

        # 如果我们想恢复训练
        if args.model_dir and hypes['train_params']['enable_resume']:
            print('-----------------Load Pretrained Model------------------')
            saved_path = os.path.join(args.model_dir, f'fold-{fold}')
            if not os.path.exists(saved_path):
                os.makedirs(saved_path)
            init_epoch, model, optimizer, lr_scheduler, scaler, f1score = train_utils.load_saved_model(saved_path,
                                                                                                       model,
                                                                                                       optimizer,
                                                                                                       lr_scheduler,
                                                                                                       scaler)
            lowest_val_epoch = init_epoch
        else:
            init_epoch = 0
            f1score = -1
            # 如果我们从头开始训练模型，我们需要创建一个文件夹去保存模型
            if origin_path is None:
                origin_path = train_utils.setup_train(hypes)
            saved_path = os.path.join(origin_path, f'fold-{fold}')
            if not os.path.exists(saved_path):
                os.makedirs(saved_path)

        # 写入内容到文件
        log_file = open(os.path.join(saved_path, 'program_output.log'), "w", encoding="utf-8")
        sys.stdout = LogWriter(sys.__stdout__, log_file)  # 保持控制台输出 + 文件记录

        model.to(device)



        if hypes['early_stop']['use']:
            early_stopping = EarlyStopping(**hypes['early_stop']['args'])


        best_f1 = f1score
        print(f"初始化f1-score: {best_f1}")
        start_time = time.time()
        continue_train = True
        for epoch in range(init_epoch, max(epochs, init_epoch)):
            if not continue_train:
                break
            # ------------------训练------------------
            # mean_loss, lr = train_one_epoch(model, optimizer, train_loader, device, epoch, num_classes,
            #                                 lr_scheduler=lr_scheduler, print_freq=args.print_freq, scaler=scaler)
            model.train()
            metric_logger = utils.MetricLogger(delimiter="  ")
            metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
            header = 'Epoch: [{}] Fold: [{}]'.format(epoch, fold)



            for images, target in metric_logger.log_every(train_loader, 1, header):
                with torch.amp.autocast(hypes['device'], enabled=scaler is not None):
                    if hypes['input_type'] == '3d':
                        # 合并一下数据
                        image, target = images.to(device), target.to(device,dtype=torch.long)
                        # image = torch.stack([modal_list[0], modal_list[1]], dim=1)
                        # image = modal_list[1].unsqueeze(1)
                    elif hypes['input_type'] == '2d':
                        image, target = images.unsqueeze(1).to(device), target.to(device,
                                                                                  dtype=torch.long)

                    output = model(image)
                    # torch.where(target == 2)[0].numel()
                    # loss = criterion(output['out'], target, loss_weight, num_classes=num_classes, ignore_index=255)
                    loss = get_loss(criterion, output['out'], target, loss_weight, num_classes=num_classes,
                                    ignore_idx=255)
                    if hypes['model']['core_method'] == 'MedNeXt' and hypes['model']['args'][
                        'deep_supervision'] == True:
                        # 下采样倍率列表
                        scale_factors = [1 / 2, 1 / 4, 1 / 8, 1 / 16]
                        aux_weight = [1 / 2, 1 / 4, 1 / 8, 1 / 16]
                        # 对每个倍率进行下采样
                        for idx, scale in enumerate(scale_factors):
                            # 计算目标尺寸
                            target_size = (int(target.shape[0] * scale), int(target.shape[1] * scale))

                            # 使用nearest进行下采样, 双线性插值会有无效浮点值
                            aux_target = F.interpolate(target.unsqueeze(1).float(), scale_factor=scale, mode='nearest')

                            # loss += criterion(output[f'out{idx + 1}'], aux_target.squeeze(1).to(torch.long),
                            #                   loss_weight,
                            #                   num_classes=num_classes, ignore_index=255) * aux_weight[idx]
                            loss += get_loss(criterion, output[f'out{idx + 1}'], aux_target.squeeze(1).to(torch.long),
                                             loss_weight, num_classes=num_classes, ignore_idx=255) * aux_weight[idx]

                optimizer.zero_grad()
                if scaler is not None:
                    scaler.scale(loss).backward()
                    # 梯度裁剪（按范数裁剪）, 只要开始两个batch会到6, 后面全是1~5之间, 所以选择5
                    # (如果使用模型初始化开始就是10多, 测试后发现加上梯度裁剪对模型收敛有坏处, 注释)
                    # 参数更新量 ≈ 学习率 × 梯度范数
                    if is_use_grad_norm:
                        torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                       max_norm=hypes['grad_norm']['args']['max_norm'])
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    # 梯度裁剪（按范数裁剪）
                    if is_use_grad_norm:
                        torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                       max_norm=hypes['grad_norm']['args']['max_norm'])
                    optimizer.step()
                # print(f"当前梯度: {calculate_grad_norm(model)}")
                if hypes['lr_scheduler']['step_per_batch']:
                    lr_scheduler.step()
                lr = optimizer.param_groups[0]["lr"]
                # 更新字段的值
                metric_logger.update(loss=loss.item(), lr=lr)
            if not hypes['lr_scheduler']['step_per_batch']:
                lr_scheduler.step()
            torch.cuda.empty_cache()  # 清空PyTorch的CUDA缓存
            gc.collect()  # 触发Python垃圾回收
            # mean_loss = metric_logger.meters["loss"].global_avg
            # --------------------------------------

            # ----------------------------验证---------------------------------------
            print("当前时间:", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            if epoch % hypes['train_params']['eval_freq'] == 0 and epoch != 0:
                model.eval()
                # calculator = Calculator(num_classes=num_classes)
                calculator = ConfusionMatrixMetric(num_classes=num_classes)
                # metrics = MetricsCalculator()
                metric_logger = utils.MetricLogger(delimiter="  ")
                header = 'Test:'
                val_loss = []
                with torch.no_grad():
                    for images, target in metric_logger.log_every(val_loader, 100, header):
                        if hypes['input_type'] == '3d':
                            # 合并一下数据
                            image, target = images.to(device), target.to(device,dtype=torch.long)
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

                        val_loss.append(float(get_loss(criterion, output, target, loss_weight, num_classes=num_classes,
                                                 ignore_idx=255).detach().cpu()))
                        if torch.isnan(output).any() or torch.isinf(output).any():
                            print("输出数据包含 NaN 或无穷大")
                            output = torch.where(torch.isnan(output), torch.full_like(output, 0), output)
                        calculator.update(output.argmax(1).detach().cpu(), target.cpu().detach().cpu())
                        # metrics.update(output.argmax(1), target)
                del output, image, images, target
                torch.cuda.empty_cache()
                gc.collect()  # 触发Python垃圾回收



                val_info = str(calculator)
                # val_info_cup = str(cup_calculator)
                print(f'CAVF info: \n {val_info}')
                print(f"validate loss: {torch.tensor(val_loss, dtype=torch.float32).mean().item():.3f}")
                print('验证：\n')
                # print(str(metrics))
                all_f1_scores = calculator.compute_dict['f1_score']

                if hypes['early_stop']['use']:
                    # 使用早停法
                    early_stopping(sum(all_f1_scores[1:]).item())
                    if early_stopping.early_stop:
                        print("Early stopping")
                        continue_train = False  # 跳出迭代，结束训练
                f1_value = sum(all_f1_scores[1:]).item()
                del calculator

            # ----------------保存模型参数-------------------
            save_file = {"model": model.state_dict(),
                         "optimizer": optimizer.state_dict(),
                         "lr_scheduler": lr_scheduler.state_dict(),
                         "epoch": epoch,
                         "args": args,
                         "best_f1": best_f1}
            if hypes['amp']:
                save_file["scaler"] = scaler.state_dict()


            if best_f1 < f1_value and epoch % hypes['train_params']['eval_freq'] == 0 and epoch != 0:
                best_f1 = f1_value
                save_file['best_f1'] = best_f1
                torch.save(save_file, os.path.join(saved_path, 'net_epoch_bestval_at%d.pth' % (epoch + 1)))
                if lowest_val_epoch != -1 and os.path.exists(os.path.join(saved_path,
                                                                          'net_epoch_bestval_at%d.pth' % (
                                                                                  lowest_val_epoch))):
                    os.remove(os.path.join(saved_path,
                                           'net_epoch_bestval_at%d.pth' % lowest_val_epoch))
                lowest_val_epoch = epoch + 1

            if epoch % hypes['train_params']['save_freq'] == 0:
                torch.save(save_file, os.path.join(saved_path, 'net_epoch%d.pth' % (epoch + 1)))
            del save_file
            torch.cuda.empty_cache()  # 清空PyTorch的CUDA缓存
            gc.collect()  # 触发Python垃圾回收
            # ------------------------------------------------
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print("training time {}".format(total_time_str))

        # 删除第一个模型并释放资源
        del model, optimizer,metric_logger  # 删除对象
        del train_loader, val_loader,train_dataset,val_dataset,scaler, lr_scheduler, criterion
        torch.cuda.empty_cache()  # 清空PyTorch的CUDA缓存
        gc.collect()  # 触发Python垃圾回收

    if args.immediately_inference:
        print(f'开始进行推理: {hypes["name"]} {hypes["dataset"]["method"]}')
        inference_main(args, hypes)


def get_model_dir_path(idx=0):
    paths = [
        [
            '../logs/compare/6mm/UNet',
            '../logs/compare/6mm/AttentionUNet',
            '../logs/compare/6mm/AVNet',
            '../logs/compare/6mm/AGNet',
            '../logs/compare/6mm/CENet',
            '../logs/compare/6mm/MNet',
            '../logs/compare/6mm/TransUNet',
            '../logs/compare/6mm/UCTransNet',
            '../logs/compare/6mm/IPN',
            '../logs/compare/6mm/IPNv2',
            '../logs/compare/6mm/H2CNet',
            '../logs/compare/6mm/SFDFormer',
        ],
        [
            # '../logs/compare/3mm/UNet',
            # '../logs/compare/3mm/AttentionUNet',
            # '../logs/compare/3mm/AVNet',
            # '../logs/compare/3mm/AGNet',
            # '../logs/compare/3mm/CENet',
            # '../logs/compare/3mm/MNet',
            # '../logs/compare/3mm/TransUNet',
            # '../logs/compare/3mm/UCTransNet',
            # '../logs/compare/3mm/IPN',
            # '../logs/compare/3mm/IPNv2',
            # '../logs/compare/3mm/H2CNet',
            # '../logs/compare/3mm/TransUNet',
            # '../logs/compare/6mm/SFDFormer',
            '../logs/compare/3mm/SFDFormer',
        ]
    ]

    return paths[idx]


def modify_config(hypes):
    """
    保险
    Args:
        hypes: 配置文件对象

    Returns: 配置文件对象

    """
    hypes['amp'] = True
    # hypes['train_params']['batch_size'] = 1
    hypes['train_params']['train_fold_list'] = [1]
    hypes['train_params']['epoches'] = 1
    hypes['dataset']['train_expand_rate'] = 1
    hypes['num_workers'] = 0
    # if hypes['dataset']['method'] == 'EyeOcdDataset':
    #     hypes['dataset']['root_dir'] = "D:\\F\\视杯视盘分割\\EYE-OCD"  # F:\\眼底图像分割\\DRIVE
    #     hypes['dataset']['train_expand_rate'] = 2
    #     hypes['train_params']['epochs'] = 80
    #     hypes['train_params']['save_freq'] = 160
    #     hypes['train_params']['train_fold_list'] = [1, 2, 3, 4, 5]
    #     hypes['postprocess']['threshold'] = 5
    #     hypes['early_stop']['use'] = True
    #     hypes['early_stop']['args']['patience'] = 20
        # if hypes['name'] == 'TransUNet':
        #     hypes['optimizer']['lr'] = 0.0001

    return hypes


if __name__ == '__main__':
    use_queue_train = True  # 是否启用队列训练
    # setup_seed()
    if not use_queue_train:
        print('-----------------Analyze Config File------------------')
        args = parse_args()
        hypes = yaml_utils.load_yaml(args.hypes_yaml, args)
        hypes['amp'] = True
        # hypes['num_workers'] = 4
        # hypes['train_params']['batch_size'] = 1
        # hypes['optimizer']['lr'] = 0.001
        main(args, hypes)
    else:
        device = 'cuda'  # 7 6 4 3
        model_dir_path = get_model_dir_path(1)
        for path in model_dir_path:
            print('-----------------Analyze Config File------------------')
            args = parse_args()
            args.model_dir = path
            hypes = yaml_utils.load_yaml(args.hypes_yaml, args)
            print(f'当前训练模型路径: {os.path.abspath(path)}')
            if device is not None:
                hypes['device'] = device
            # hypes['train_params']['train_fold_list'] = [1, 2, 3, 4, 5]
            # hypes['train_params']['epochs'] = 40
            # hypes['num_workers'] = 4
            # hypes = modify_config(hypes)
            # hypes['train_params']['epochs'] = 1  # 测试
            # hypes['dataset']['train_expand_rate'] = 1  # 测试
            # hypes['train_params']['train_fold_list'] = [1]  # 测试
            main(args, hypes)
