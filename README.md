# OCD-Seg2D(用于2D视杯视盘联合分割)

## 框架亮点：
* 配置即模型
* 实现了5折交叉检验
* 支持多种指标计算, 如F1-score, Dice, Mean Dice, IOU, AUC, ROC
* 集成nnUnet的多种Trick, 如Early Stop, 滑动窗口预测, 多项式学习率策略, 数据增强, 基于阈值的联通量过滤后处理
* 简单的修改即可添加自定义项, 如模型, 学习率策略, 损失函数

## 环境配置：
* Python 3.8/3.9/3.10
* Pytorch 1.13.1
* Ubuntu或Centos(Windows暂不支持多GPU训练)
* 最好使用GPU训练
* 详细环境配置见`requirements.txt`

## 文件结构：
```
  ├── data_utils: 数据读取与数据增强
  ├── hypes_yaml: 配置文件
  ├── logs: 训练与验证过程, 数据保存位置
  ├── loss: 损失函数
  ├── lr_schedular: 学习率策略
  ├── metric: 计算指标
  ├── models: 存放所有的模型
  ├── tools: 需要执行的脚本文件, 如训练, 验证, 5折交叉检验数据划分, 数据集计算
  └── utils: 工具类
```

## 视网膜血管分割数据集下载地址：
* 官网地址： [https://drive.grand-challenge.org/](https://drive.grand-challenge.org/)
* 百度云链接： [https://pan.baidu.com/s/1Tjkrx2B9FgoJk0KviA-rDw](https://pan.baidu.com/s/1Tjkrx2B9FgoJk0KviA-rDw)  密码: 8no8


## 训练/验证 方法
* 确保提前准备好数据集
* 若要使用单GPU或者CPU训练，直接使用tools/train.py训练脚本
* 若要使用单GPU或者CPU验证，直接使用tools/inference.py训练脚本
```shell
# 训练
python tools/train.py --hypes_yaml ${CONFIG_FILE} [--model_dir  ${CHECKPOINT_FOLDER}]
# exp: python tools/train.py --hypes_yaml ../hypes_yaml/config.yaml
# exp: python tools/train.py --model_dir ../logs/test
```

```shell
# 验证
python tools/inference.py --model_dir  ${CHECKPOINT_FOLDER}
# exp: python tools/inference.py --model_dir ../logs/test
```

```shell
# 训练
nohup python train.py  2>&1 > /tmp/JY-MedSeg-2D/log/drive-20250113.log &
nohup python train.py  2>&1 > /tmp/JY-MedSeg-2D/log/chase-20250113.log &
nohup python train.py  2>&1 > /tmp/JY-MedSeg-2D/log/hrf-20250113.log &
nohup python train.py  2>&1 > /tmp/JY-MedSeg-2D/log/stare-20250113.log &
nohup python train.py  2>&1 > /tmp/JY-MedSeg-2D/log/eyeseg-20250113.log &
tail -f /tmp/JY-MedSeg-2D/log/drive-20250113.log

# 推理
nohup python tools/inference.py  2>&1 > /tmp/JY-MedSeg-2D/log/drive-20250113-inference.log &
nohup python tools/inference.py  2>&1 > /tmp/JY-MedSeg-2D/log/chase-20250113-inference.log &
nohup python tools/inference.py  2>&1 > /tmp/JY-MedSeg-2D/log/hrf-20250113-inference.log &
nohup python tools/inference.py  2>&1 > /tmp/JY-MedSeg-2D/log/stare-20250113-inference.log &
```

## 注意事项
* 请在训练前执行 [kfold_split.py](tools%2Fkfold_split.py) 进行5折交叉检验数据划分
* 请在训练前执行 [compute_mean_std.py](tools%2Fcompute_mean_std.py) 得到数据集的标准差和方差


## 实验设置

模型 UNet MNet CENet AGNet AttentionUnet U2Net TransUNet DenseUNet MISSFormer IterNet



