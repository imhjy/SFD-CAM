# SFD-CAM

## 数据集

使用OCTA-500数据集

## 框架亮点：
* 配置即模型
* 实现了5折交叉检验
* 支持多种指标计算, 如F1-score, Dice, Mean Dice, IOU, AUC, ROC
* 集成nnUnet的多种Trick, 如Early Stop, 滑动窗口预测, 多项式学习率策略, 数据增强, 基于阈值的联通量过滤后处理
* 简单的修改即可添加自定义项

## 环境配置：
* Python 3.10
* Pytorch 1.13.1
* Ubuntu或Centos(Windows暂不支持多GPU训练)
* 最好使用GPU训练



## 训练/验证 方法
* 确保提前准备好数据集
* 若要使用单GPU或者CPU训练，直接使用tools/train.py训练脚本
* 若要使用单GPU或者CPU验证，直接使用tools/inference.py训练脚本
```shell
# 训练
python tools/train.py --hypes_yaml ${CONFIG_FILE} [--model_dir  ${CHECKPOINT_FOLDER}]
```

```shell
# 验证
python tools/inference.py --model_dir  ${CHECKPOINT_FOLDER}
# exp: python tools/inference.py --model_dir ../logs/test
```

```shell
# 训练
nohup python train.py  2>&1 > /tmp/xxxxx/log/drive-20251013.log &
```





