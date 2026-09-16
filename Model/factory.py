"""项目所有算法共用的模型构造入口。"""

from Model.cnn import CNN
from Model.resnet import ResNet_PC


def build_model(args, num_classes=None):
    """根据命令行配置创建模型，统一返回 ``(features, logits)`` 接口。"""
    num_classes = args.num_classes if num_classes is None else num_classes
    if args.model == "cnn":
        return CNN(
            input_channels=3,
            num_classes=num_classes,
            feature_dim=args.cnn_feature_dim,
            dataset_name=args.dataset,
        )
    if args.model == "resnet":
        return ResNet_PC(
            resnet_size=8,
            scaling=4,
            save_activations=False,
            group_norm_num_groups=None,
            freeze_bn=False,
            freeze_bn_affine=False,
            num_classes=num_classes,
        )
    raise ValueError(f"不支持的模型：{args.model}")
