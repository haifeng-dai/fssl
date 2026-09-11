import logging

import numpy as np
from torch.utils.data.dataset import Dataset
from torchvision import transforms

from .randaugment import RandAugmentMC


def classify_label(dataset, num_classes: int):
    list1 = [[] for _ in range(num_classes)]
    for idx, datum in enumerate(dataset):
        list1[datum[1]].append(idx)
    return list1


def show_clients_data_distribution(
    dataset, clients_indices_labeled, clients_indices_unlabeled, num_classes
):
    dict_per_client_labeled = []
    dict_per_client_unlabeled = []

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


def partition_train(list_label2indices: list, ipc):

    list_label2indices_labeled = []
    list_label2indices_unlabeled = []

    for indices in list_label2indices:
        idx_shuffle = np.random.permutation(indices)

        list_label2indices_labeled.append(idx_shuffle[:ipc])
        list_label2indices_unlabeled.append(idx_shuffle[ipc:])
    return list_label2indices_labeled, list_label2indices_unlabeled


class Indices2Dataset_labeled(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.indices = None
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
        self.indices = indices

        self.client_dataset = [self.dataset[i] for i in indices]
        self.client_dataset *= 2000
        # 因为使用batch 128时，每次epoch都需要重新 iter(dataset) 一次，每次100ms
        # 这里复制多次dataset，减少运行 iter 函数的次数
        # 数字是随便定的

    def __getitem__(self, idx):
        image, label = self.client_dataset[idx]
        image = self.label_trans(image)
        return image, label

    def __len__(self):
        return len(self.client_dataset)


class Indices2Dataset_unlabeled_fixmatch(Dataset):
    def __init__(self, dataset):
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
        self.indices = indices

        self.client_dataset = [self.dataset[i] for i in self.indices]
        self.client_dataset_len = len(self.client_dataset)
        self.client_dataset *= 50  # save time loading data

    def fixmatch(self, image):
        weak = self.weak(image)
        strong = self.strong(image)
        return self.normalize(weak), self.normalize(strong)

    def __getitem__(self, idx):

        image, label = self.client_dataset[idx]

        image1, image2 = self.fixmatch(image)
        return image1, image2, label

    def __len__(self):
        return self.client_dataset_len
