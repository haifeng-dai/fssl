"""轻量 CNN，统一提供联邦训练所需的特征与分类接口。"""

from torch import nn


class CNN(nn.Module):
    def __init__(
        self, input_channels=3, num_classes=10, feature_dim=512, dataset_name="cifar10"
    ):
        super().__init__()
        dataset_name = dataset_name.lower()
        if dataset_name in {"mnist", "fashionmnist", "femnist", "emnist"}:
            flattened_dim = 64 * 7 * 7
        elif dataset_name == "tiny_imagenet":
            flattened_dim = 64 * 16 * 16
        elif dataset_name in {"cars", "flowers102"}:
            flattened_dim = 64 * 56 * 56
        else:
            flattened_dim = 64 * 8 * 8

        self.dim = feature_dim
        self.extractor = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(flattened_dim, feature_dim),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(feature_dim, num_classes)
        # ProxyFL 依赖这两个投影层；其他算法不会使用它们。
        self.feat_proj = nn.Linear(feature_dim, feature_dim)
        self.proxy_proj = nn.Linear(feature_dim, feature_dim, bias=False)

    def forward(self, x):
        features = self.extractor(x)
        return features, self.classifier(features)
