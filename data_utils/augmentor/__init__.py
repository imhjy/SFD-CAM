from data_utils.augmentor.augment import get_enhance_gray_transformer, get_nnunet_transformer, get_octa500_transformer, \
    get_octa500_2d_transformer

__all__ = {
    'EnhanceGray': get_enhance_gray_transformer,
    'nnunet': get_nnunet_transformer,
    'OCTA500': get_octa500_transformer,
    'OCTA5002d': get_octa500_2d_transformer,
}


def build_dataset_transformer(hypes, train=True):
    """
    创建数据集对应的增强器
    Args:
        hypes: 配置文件字典
        train: 是否是训练模型

    Returns:
        需要的数据集
    """

    dataset_augmentor = __all__[hypes['augmentor']['core_method']](
        train=train,
        hypes=hypes
    )

    return dataset_augmentor
