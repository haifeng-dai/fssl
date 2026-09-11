import math

from torch import nn


def norm2d(group_norm_num_groups, planes):
    if group_norm_num_groups is not None and group_norm_num_groups > 0:
        return nn.GroupNorm(group_norm_num_groups, planes)
    else:
        return nn.BatchNorm2d(planes)


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, out_planes, stride=1, downsample=None, group_norm_num_groups=None):
        super().__init__()
        self.conv1 = conv3x3(in_planes, out_planes, stride)
        self.bn1 = norm2d(group_norm_num_groups, out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_planes, out_planes)
        self.bn2 = norm2d(group_norm_num_groups, out_planes)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        return self.relu(out + residual)


class ResNet_PC(nn.Module):
    def __init__(self, resnet_size=8, scaling=4, save_activations=False,
                 group_norm_num_groups=None, freeze_bn=False, freeze_bn_affine=False, num_classes=10):
        super().__init__()
        self.freeze_bn = freeze_bn
        self.freeze_bn_affine = freeze_bn_affine
        self.save_activations = save_activations
        self.num_classes = num_classes
        self.scaling = scaling

        if resnet_size % 6 != 2:
            raise ValueError("resnet_size must be 6n + 2:", resnet_size)
        block_nums = (resnet_size - 2) // 6

        planes1, planes2, planes3 = int(16 * scaling), int(32 * scaling), int(64 * scaling)

        self.conv1 = nn.Conv2d(3, planes1, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm2d(group_norm_num_groups, planes1)
        self.relu = nn.ReLU(inplace=True)

        self.inplanes = planes1
        self.layer1 = self._make_block(planes1, block_nums, group_norm_num_groups)
        self.layer2 = self._make_block(planes2, block_nums, stride=2, group_norm_num_groups=group_norm_num_groups)
        self.layer3 = self._make_block(planes3, block_nums, stride=2, group_norm_num_groups=group_norm_num_groups)

        self.avgpool = nn.AvgPool2d(kernel_size=8)
        self.classifier = nn.Linear(planes3, num_classes)
        self.dim = planes3
        self.feat_proj = nn.Linear(self.dim, self.dim)
        self.proxy_proj = nn.Linear(self.dim, self.dim, bias=False)

        self._init_weights()

    def _make_block(self, planes, block_num, stride=1, group_norm_num_groups=None):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False),
                norm2d(group_norm_num_groups, planes),
            )
        layers = [BasicBlock(self.inplanes, planes, stride, downsample, group_norm_num_groups)]
        self.inplanes = planes
        for _ in range(1, block_num):
            layers.append(BasicBlock(self.inplanes, planes, group_norm_num_groups=group_norm_num_groups))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_bn:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    if self.freeze_bn_affine:
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False
        return self

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))

        x = self.layer1(x)
        activation1 = x
        x = self.layer2(x)
        activation2 = x
        x = self.layer3(x)
        activation3 = x

        x = self.avgpool(x).flatten(1)
        feature = x
        y = self.classifier(x)

        if self.save_activations:
            self.activations = [activation1, activation2, activation3]

        return feature, y
