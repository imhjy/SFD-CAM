from data_utils.augmentor import build_dataset_transformer
from data_utils.datasets.EyeOcdDataset import EyeOcdDataset
from data_utils.datasets.RefugeDataset import RefugeDataset
from data_utils.datasets.DrishtiDataset import DrishtiDataset
from data_utils.datasets.GammaDataset import GammaDataset
from data_utils.datasets.RimOneDataset import RimOneDataset
from data_utils.datasets.OrigaDataset import OrigaDataset
from data_utils.datasets.OCTA500Dataset import OCTA500Dataset

__all__ = {
    'EyeOcdDataset': EyeOcdDataset,
    'RefugeDataset': RefugeDataset,
    'DrishtiDataset': DrishtiDataset,
    'GammaDataset': GammaDataset,
    'RimOneDataset': RimOneDataset,
    'OrigaDataset': OrigaDataset,
    'OCTA500Dataset': OCTA500Dataset,
}


def build_dataset(hypes, train=True, fold=0):
    """
    创建数据集
    Args:
        hypes: 配置文件字典
        train: 是否是训练模型
        fold: 使用哪一折, fold=0表示默认
    Returns:
        需要的数据集
    """

    dataset_name = hypes['dataset']['method']
    error_message = f"{dataset_name} 没有找到. " \
                    f"请将数据集添加到: " \
                    f"data_utils/datasets/init.py"
    assert dataset_name in list(__all__), error_message

    dataset = __all__[dataset_name](
        hypes=hypes,
        train=train,
        transforms=build_dataset_transformer(hypes=hypes, train=train),
        fold=fold
    )

    return dataset
