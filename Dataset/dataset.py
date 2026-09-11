import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms

from .randaugment import RandAugmentMC


def classify_label(dataset, num_classes: int):
    """按类别收集数据集样本下标。"""
    list1 = [[] for _ in range(num_classes)]
    for idx, datum in enumerate(dataset):
        list1[datum[1]].append(idx)
    return list1


def partition_train(list_label2indices: list, ipc):
    """从每个类别中随机抽取 ipc 个样本，其余样本作为无标签数据。"""

    list_label2indices_labeled = []
    list_label2indices_unlabeled = []

    # 每个类别独立打乱，保证抽取过程不受原始顺序影响。
    for indices in list_label2indices:
        idx_shuffle = np.random.permutation(indices)

        list_label2indices_labeled.append(idx_shuffle[:ipc])
        list_label2indices_unlabeled.append(idx_shuffle[ipc:])
    return list_label2indices_labeled, list_label2indices_unlabeled


def show_clients_data_distribution(
    dataset, clients_indices_labeled, clients_indices_unlabeled, num_classes
):
    """统计并打印每个客户端的有标签和无标签类别分布。"""
    dict_per_client_labeled = []
    dict_per_client_unlabeled = []

    # 逐个客户端统计两类数据的标签数量。
    for client, indices in enumerate(
        zip(clients_indices_labeled, clients_indices_unlabeled)
    ):
        nums_data_labeled = [0 for _ in range(num_classes)]
        nums_data_unlabeled = [0 for _ in range(num_classes)]
        idx_labeled, idx_unlabeled = indices

        for idx in idx_labeled:
            label = dataset[idx][1]
            nums_data_labeled[label] += 1
        dict_per_client_labeled.append(nums_data_labeled)

        for idx in idx_unlabeled:
            label = dataset[idx][1]
            nums_data_unlabeled[label] += 1
        dict_per_client_unlabeled.append(nums_data_unlabeled)

        print(f"client {client} labeled number per class : {nums_data_labeled}")
        print(f"client {client} unlabeled number per class  : {nums_data_unlabeled}")

    return dict_per_client_labeled, dict_per_client_unlabeled


class SharedImageDataset(Dataset):
    """从 CPU 共享内存中按索引读取原始图片和标签。"""

    def __init__(self, images, labels):
        """保存共享图片张量和共享标签张量，不复制底层数据。"""
        self.images = images
        self.labels = labels

    def __getitem__(self, index):
        """将共享内存中的单张图片转换为 PIL 图片并返回标签。"""
        image = Image.fromarray(self.images[index].numpy())
        return image, int(self.labels[index].item())

    def __len__(self):
        """返回共享数据集中的样本数。"""
        return self.images.shape[0]


class Indices2Dataset_labeled(Dataset):
    """客户端有标签数据视图，只保存全局数据集下标。"""

    def __init__(self, dataset, repeat=2000):
        """创建有标签视图，并配置随机增强和归一化操作。"""
        self.dataset = dataset
        self.indices = None
        self.repeat = repeat
        self.label_trans = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(
                    size=32, padding=int(32 * 0.125), padding_mode="reflect"
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616)
                ),
            ]
        )

    def load(self, indices: list):
        """加载当前客户端的样本下标，不复制图片对象。"""
        # 只保存索引。旧实现会将每张图片复制数千次，多 GPU worker 会显著放大内存占用。
        self.indices = list(indices)

    def __getitem__(self, idx):
        """按循环下标取样，并执行一次有标签数据增强。"""
        image, label = self.dataset[self.indices[idx % len(self.indices)]]
        image = self.label_trans(image)
        return image, label

    def __len__(self):
        """返回逻辑长度；repeat 只扩大采样次数，不扩大内存占用。"""
        return len(self.indices) * self.repeat if self.indices else 0


class Indices2Dataset_unlabeled_fixmatch(Dataset):
    """客户端无标签数据视图，为同一图片生成弱增强和强增强样本。"""

    def __init__(self, dataset):
        """创建弱增强、强增强和归一化变换。"""
        self.dataset = dataset
        self.indices = None
        self.weak = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(
                    size=32, padding=int(32 * 0.125), padding_mode="reflect"
                ),
            ]
        )
        self.strong = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(
                    size=32, padding=int(32 * 0.125), padding_mode="reflect"
                ),
                RandAugmentMC(n=2, m=10),
            ]
        )
        self.normalize = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616)
                ),
            ]
        )

    def load(self, indices: list):
        """加载当前客户端的无标签样本下标。"""
        # 与有标签数据相同，不为每个客户端复制图片对象。
        self.indices = list(indices)
        self.client_dataset_len = len(self.indices)

    def fixmatch(self, image):
        """对同一原始图片分别执行弱增强和强增强。"""
        weak = self.weak(image)
        strong = self.strong(image)
        return self.normalize(weak), self.normalize(strong)

    def __getitem__(self, idx):
        """读取一个样本并返回弱增强图、强增强图及其真实标签。"""
        image, label = self.dataset[self.indices[idx]]

        image1, image2 = self.fixmatch(image)
        return image1, image2, label

    def __len__(self):
        """返回当前客户端无标签视图的逻辑长度。"""
        return self.client_dataset_len
