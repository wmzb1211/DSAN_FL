# 划分数据到各客户端（非IID支持）
import numpy as np
from torch.utils.data import Subset

def split_iid(dataset, num_clients):
    """IID数据划分"""
    num_samples = len(dataset) // num_clients
    indices = np.random.permutation(len(dataset))
    return [
        Subset(dataset, indices[i*num_samples:(i+1)*num_samples])
        for i in range(num_clients)
    ]


def split_non_iid(dataset, num_clients, shards_per_client=2):
    """
    非独立同分布数据划分 (每个客户端分配2个类别的数据)
    实现参考: https://arxiv.org/abs/1902.01046 (FedAvg论文的Non-IID方法)

    参数:
        dataset: 原始数据集 (必须包含 targets 属性)
        num_clients: 客户端数量
        shards_per_client: 每个客户端分配的碎片数 (默认2)

    返回:
        List[Subset]: 划分后的客户端数据列表
    """
    # 获取数据类别和索引
    targets = np.array(dataset.targets)
    num_classes = len(np.unique(targets))

    # 计算总碎片数 (每个类别的样本分成 2*num_clients 个碎片)
    shards_per_class = 2 * num_clients // num_classes
    total_shards = shards_per_class * num_classes

    # 按类别排序索引
    sorted_indices = np.argsort(targets)
    class_indices = [sorted_indices[targets[sorted_indices] == i] for i in range(num_classes)]

    # 为每个类别生成碎片
    shards = []
    for class_idx in range(num_classes):
        class_data = class_indices[class_idx]
        split_points = np.linspace(0, len(class_data), shards_per_class + 1, dtype=int)
        for i in range(shards_per_class):
            shard = class_data[split_points[i]:split_points[i + 1]]
            shards.append(shard)

    # 随机分配碎片给客户端
    np.random.shuffle(shards)
    client_shards = np.array_split(shards, num_clients)

    # 合并每个客户端的碎片索引
    client_indices = []
    for shard_group in client_shards:
        indices = np.concatenate([shard for shard in shard_group])
        client_indices.append(indices)

    return [Subset(dataset, indices) for indices in client_indices]