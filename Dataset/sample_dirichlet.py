import functools
import math

import numpy as np


def clients_indices_homo(list_label2indices: list, num_classes: int, num_clients: int):
    partition_list = []
    for index_class in range(num_classes):
        class_partition_list = []
        class_list_length = len(list_label2indices[index_class])
        for client in range(num_clients):
            class_partition = list_label2indices[index_class][
                math.floor(client / num_clients * class_list_length) : math.floor(
                    (client + 1) / num_clients * class_list_length
                )
            ]

            class_partition_list.append(class_partition)
        partition_list.append(class_partition_list)

    list_label2indices = []
    for client in range(num_clients):
        client_partition_list = []
        for class_idx in range(num_classes):
            client_partition_list.extend(partition_list[class_idx][client])
        list_label2indices.append(np.array(client_partition_list))

    return list_label2indices


def clients_indices(
    list_label2indices: list,
    num_classes: int,
    num_clients: int,
    non_iid_alpha: float,
    seed=None,
):
    indices2targets = []

    for label, indices in enumerate(list_label2indices):
        for idx in indices:
            indices2targets.append((idx, label))

    batch_indices = build_non_iid_by_dirichlet(
        seed=seed,
        indices2targets=indices2targets,
        non_iid_alpha=non_iid_alpha,
        num_classes=num_classes,
        num_indices=len(indices2targets),
        n_workers=num_clients,
    )

    indices_dirichlet = functools.reduce(lambda x, y: x + y, batch_indices)

    list_client2indices = partition_balance(indices_dirichlet, num_clients)

    return list_client2indices


def build_non_iid_by_dirichlet(
    seed, indices2targets, non_iid_alpha, num_classes, num_indices, n_workers
):
    # 1. 初始化独立的 NumPy 随机数生成器，确保数据集划分结果在相同种子下完全可复现
    random_state = np.random.RandomState(seed)

    # 2. 辅助工作组大小（n_auxi_workers = 2）：
    # 采用分治策略，固定将 2 个客户端打包为一个采样小组进行数据瓜分。
    # 目的：避免总客户端数较多时狄利克雷分布维度过高，导致单类别在某些客户端上分配极度稀疏甚至退化。
    n_auxi_workers = 2
    # 断言校验：小组大小不能超过参与划分的总客户端数量
    assert n_auxi_workers <= n_workers

    # 3. 全局随机打乱 (样本索引, 类别标签) 元组列表，打乱原始数据集中的样本排列顺序
    random_state.shuffle(indices2targets)

    # 4. 将全部样本均分为多个子数据集，与辅助小组一一对应：
    # from_index: 初始化滑动切片的起始游标，从第 0 个样本开始
    from_index = 0
    # splitted_targets: 容器列表，用于按顺序存放所有切分出的小组子数据集
    splitted_targets = []
    # num_splits: 计算需要切分的总数据块数量 = ceil(总客户端数 / 每组客户端数)
    # 例如 20 个客户端，每组 2 人，则切为 ceil(20/2) = 10 个子数据集
    num_splits = math.ceil(n_workers / n_auxi_workers)

    # 循环切分数据块，每一块数据供一个 2 人小组独立瓜分
    for idx in range(num_splits):
        # 计算当前子数据块的结束下标：
        # (n_auxi_workers / n_workers) 是当前小组人数占总人数的比例，乘以 num_indices 得到本小组应得的标准样本步长
        to_index = from_index + int(n_auxi_workers / n_workers * num_indices)

        # 截取样本区间并存入列表。
        # 边界保护：若为最后一个分块（idx == num_splits - 1），右边界强制取 num_indices（数据集终点），
        # 吸收前面整除取整损失的零头样本，确保整个数据集不重不漏、一个样本都不丢失
        splitted_targets.append(
            indices2targets[
                from_index : (num_indices if idx == num_splits - 1 else to_index)
            ]
        )
        # 将滑动游标右移，使下一个数据块的起点紧接当前数据块的终点
        from_index = to_index

    # idx_batch: 最终容器，存放所有客户端分配到的样本全局索引（长度为总客户端数）
    idx_batch = []

    # 5. 遍历每个子数据块，在对应的小组客户端（通常为 2 个）内部进行狄利克雷 Non-IID 分配
    for _targets in splitted_targets:
        # 将当前子数据块转为二维 NumPy 数组：第 0 列为样本全局索引，第 1 列为类别标签
        _targets = np.array(_targets)
        # 获取当前子数据集中的样本总数量
        _targets_size = len(_targets)

        # 确定分配给当前子数据块的客户端数量（一般为 2，剩余不足 2 时取剩余客户端数）
        _n_workers = min(n_auxi_workers, n_workers)
        # 从未分配的总客户端数中扣除当前小组所消耗的客户端数
        n_workers = n_workers - n_auxi_workers

        # 初始化单客户端最小样本数量变量，用于检测采样质量
        min_size = 0
        _idx_batch = None

        # 6. 拒绝采样（质量控制循环）：
        # 只要当前小组内分配到样本最少的客户端样本数 < 该小组平均样本数的 50%，
        # 说明本次狄利克雷采样过于极端（某客户端几乎饿死），推倒整个小组的采样重新执行
        while min_size < int(0.50 * _targets_size / _n_workers):
            # 为当前小组内的每一个客户端初始化一个空的样本索引接收列表
            _idx_batch = [[] for _ in range(_n_workers)]

            # 逐个类别分别从狄利克雷分布中采样分配比例
            for _class in range(num_classes):
                # 找出当前子数据集中真实标签属于 _class 的所有行下标
                idx_class = np.where(_targets[:, 1] == _class)[0]
                # 根据行下标提取出原始样本的实际全局索引
                idx_class = _targets[idx_class, 0]

                try:
                    # 7. 从对称狄利克雷分布中采样分配比例向量，长度为当前小组人数 _n_workers
                    # 参数 non_iid_alpha 越小（如 0.1），采样出的分布比例越极端偏斜（Non-IID 程度越高）
                    proportions = random_state.dirichlet(
                        np.repeat(non_iid_alpha, _n_workers)
                    )

                    # 8. 动态容量上限控制：
                    # 检查各客户端当前累计已分配的样本数是否已达到该组的平均样本量。
                    # 若某个客户端已经拿够了平均数额，强制将其本轮分配比例置为 0，防止某个客户端被撑得过大
                    proportions = np.array(
                        [
                            p * (len(idx_j) < _targets_size / _n_workers)
                            for p, idx_j in zip(proportions, _idx_batch)
                        ]
                    )

                    # 对过滤后的比例向量重新归一化，使其总和恢复为 1.0
                    proportions = proportions / proportions.sum()

                    # 将分配比例转换为累积样本数整数切分点：
                    # 例如 proportions=[0.3, 0.7]，当前类别有 10 个样本，cumsum*10 得到 [3, 10]，[:-1] 得到分割坐标 [3]
                    proportions = (np.cumsum(proportions) * len(idx_class)).astype(int)[
                        :-1
                    ]

                    # 利用 np.split 在分割坐标处将当前类别的样本索引切段，并分别追加给当前小组的各个客户端
                    _idx_batch = [
                        idx_j + idx.tolist()
                        for idx_j, idx in zip(
                            _idx_batch, np.split(idx_class, proportions)
                        )
                    ]

                    # 统计当前小组中每个客户端已获得的总样本数
                    sizes = [len(idx_j) for idx_j in _idx_batch]
                    # 更新当前小组内获得样本最少的客户端的样本量，用于 while 条件校验
                    min_size = min([_size for _size in sizes])

                except ZeroDivisionError:
                    # 若所有客户端均达到容量上限导致 proportions.sum() == 0 时，捕获除零异常跳过
                    pass

        # 9. 当前小组采样结果通过容量下限检验后，将该小组各客户端的样本列表追加到全局结果列表
        if _idx_batch is not None:
            idx_batch += _idx_batch

    # 返回所有客户端的分配索引列表，长度等于 n_workers
    return idx_batch


def partition_balance(idxs, num_split: int):
    # 1. 计算均分基准大小与余数：
    # num_per_part: 每个客户端分得的基础样本数（整除部分）
    # r: 无法整除剩余的余数样本（0 <= r < num_split）
    # 分配策略：前 r 个客户端每人分 (num_per_part + 1) 个样本，其余客户端每人分 num_per_part 个样本
    # 保证任意两个客户端样本总数差值最多为 1，实现绝对数量均衡
    num_per_part, r = len(idxs) // num_split, len(idxs) % num_split

    # parts: 用于存放最终每个客户端分配到的样本切片列表
    parts = []
    # i: 当前切片的起始游标，从第 0 个元素开始
    # r_used: 已经分配了额外余数 (+1) 的客户端数量计数器
    i, r_used = 0, 0

    # 2. 循环遍历并切片整个样本列表
    while i < len(idxs):
        # 若已分配额外余数名额还未用完（r_used < r）：
        # 本切片分得 (num_per_part + 1) 个样本
        if r_used < r:
            parts.append(idxs[i : (i + num_per_part + 1)])
            # 游标向后推进 (num_per_part + 1)
            i += num_per_part + 1
            # 消耗一个余数名额
            r_used += 1
        # 若余数名额已用完（r_used >= r）：
        # 剩下的客户端每个分得基准大小 num_per_part 个样本
        else:
            parts.append(idxs[i : (i + num_per_part)])
            # 游标向后推进 num_per_part
            i += num_per_part

    # 返回切分好的列表，元素总个数严格等于 num_split
    return parts


def sample_dirichlet_balanced(
    list_label2indices: list,
    num_classes: int,
    num_clients: int,
    non_iid_alpha: float,
    seed=None,
):
    """标准的数量均衡 Dirichlet 划分：

    1. 全局采样：为每个类别在全部客户端上采样一个全局 Dirichlet 比例；
    2. 严格保量：确保各客户端分配到的样本总数严格均衡 (数量均衡)；
    3. 拒绝错位：依据各类最终预算精确分配，不存在跨客户端的盲目切断和类别错位污染。
    """
    rng = np.random.RandomState(seed)

    # 每个类别的样本索引池（打乱）
    label_indices = [
        rng.permutation(indices).tolist() for indices in list_label2indices
    ]
    total_samples = sum(len(indices) for indices in label_indices)

    # 1. 严格计算每个客户端的目标总样本数（严格平衡）
    samples_per_client = total_samples // num_clients

    # 2. 构造分配容量矩阵：(num_classes, num_clients)
    # 为每一个类别，直接在所有客户端上采样一个全局 Dirichlet 分布向量
    props = np.zeros((num_classes, num_clients))
    for c in range(num_classes):
        props[c] = rng.dirichlet(np.repeat(non_iid_alpha, num_clients))

    # 3. 双向约束调整（保持各类样本用完，同时各客户端总数平衡）
    counts = np.zeros((num_classes, num_clients), dtype=int)
    for c in range(num_classes):
        class_total = len(label_indices[c])
        c_alloc = (props[c] * class_total).astype(int)
        counts[c] = c_alloc

    # 4. 微调配额：确保每个客户端的样本总数严格等于 samples_per_client
    client_totals = counts.sum(axis=0)
    for k in range(num_clients):
        diff = samples_per_client - client_totals[k]

        # 如果少了，从该客户端权重较高且还有剩余样本的类别中优先补充
        while diff > 0:
            preferred_classes = np.argsort(-props[:, k])
            allocated = False
            for c in preferred_classes:
                if len(label_indices[c]) > counts[c].sum():
                    counts[c, k] += 1
                    diff -= 1
                    allocated = True
                    break
            if not allocated:
                break

    # 5. 根据最终确定的 counts[c, k] 数量矩阵，直接切片分发真实样本下标
    client_indices = [[] for _ in range(num_clients)]
    for c in range(num_classes):
        cur_ptr = 0
        for k in range(num_clients):
            num_take = counts[c, k]
            client_indices[k].extend(label_indices[c][cur_ptr : cur_ptr + num_take])
            cur_ptr += num_take

    return [np.array(indices) for indices in client_indices]


def sample_dirichlet_natural(
    list_label2indices: list,
    num_classes: int,
    num_clients: int,
    non_iid_alpha: float,
    seed=None,
    min_size_per_client: int = 10,
):
    """标准的自然非均衡 Dirichlet 划分 (Unbalanced Dirichlet Non-IID)：

    1. 数学纯粹：直接对每个类别从 Dir(alpha) 采样并按比例分配给各客户端；
    2. 自然异构：允许各客户端样本总量不相等，模拟真实世界的设备数据量差异；
    3. 最小阈值保护：若极端偏斜导致某客户端样本过少 (< min_size_per_client)，自动重新采样。
    """
    rng = np.random.RandomState(seed)

    while True:
        client_indices = [[] for _ in range(num_clients)]

        for c in range(num_classes):
            # 取出该类别所有样本下标并打乱
            indices = rng.permutation(list_label2indices[c])

            # 从 Dirichlet 分布中采样一个属于所有客户端的比例向量
            proportions = rng.dirichlet(np.repeat(non_iid_alpha, num_clients))

            # 转换为整数切分点
            split_points = (np.cumsum(proportions) * len(indices)).astype(int)[:-1]

            # 一刀切开直接分发给每个客户端
            for k, split in enumerate(np.split(indices, split_points)):
                client_indices[k].extend(split.tolist())

        # 检查是否所有客户端的数据量都达到了安全下限，防止 DataLoader 空 batch
        sizes = [len(idx) for idx in client_indices]
        if min(sizes) >= min_size_per_client:
            break

    return [np.array(idx) for idx in client_indices]
