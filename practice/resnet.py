from __future__ import annotations

from typing import Type

import torch
import torch.nn as nn
from torch import Tensor


def conv3x3(
    in_channels: int,
    out_channels: int,
    stride: int = 1,
) -> nn.Conv2d:
    """
    3×3卷积：
    padding=1 保证 stride=1 时特征图尺寸不变。
    """
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


def conv1x1(
    in_channels: int,
    out_channels: int,
    stride: int = 1,
) -> nn.Conv2d:
    """
    1×1卷积：
    用于调整通道数，或配合 stride=2 进行下采样。
    """
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        stride=stride,
        bias=False,
    )


class BasicBlock(nn.Module):
    """
    ResNet-18、ResNet-34 使用的基本残差块。

    主分支：
        3×3卷积 -> BN -> ReLU
        3×3卷积 -> BN

    捷径分支：
        identity 或 1×1卷积

    输出：
        ReLU(F(x) + shortcut(x))
    """

    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
    ) -> None:
        super().__init__()

        self.conv1 = conv3x3(
            in_channels,
            out_channels,
            stride=stride,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(
            out_channels,
            out_channels,
            stride=1,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 当尺寸或通道数不同时，对捷径分支进行变换
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        # 主分支 F(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # 捷径分支
        if self.downsample is not None:
            identity = self.downsample(x)

        # 残差相加
        out = out + identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """
    ResNet-50、101、152 使用的瓶颈残差块。

    主分支：
        1×1卷积：降低通道
        3×3卷积：提取特征
        1×1卷积：恢复/扩展通道

    expansion=4：
        若基础通道为64，最终输出通道为64×4=256。
    """

    expansion = 4

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
    ) -> None:
        super().__init__()

        # 先使用1×1卷积降低通道
        self.conv1 = conv1x1(
            in_channels,
            out_channels,
            stride=1,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        # TorchVision风格：stride放在3×3卷积
        self.conv2 = conv3x3(
            out_channels,
            out_channels,
            stride=stride,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 通道扩展4倍
        self.conv3 = conv1x1(
            out_channels,
            out_channels * self.expansion,
            stride=1,
        )
        self.bn3 = nn.BatchNorm2d(
            out_channels * self.expansion
        )

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        # 主分支 F(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # 捷径分支
        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """
    通用ResNet框架。

    layers表示四个阶段分别包含多少个残差块。

    例如ResNet-34：
        layers = [3, 4, 6, 3]
    """

    def __init__(
        self,
        block: Type[BasicBlock] | Type[Bottleneck],
        layers: list[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        small_input: bool = False,
    ) -> None:
        super().__init__()

        self.in_channels = 64

        # 对ImageNet等大尺寸图像使用标准Stem
        if not small_input:
            self.conv1 = nn.Conv2d(
                in_channels=3,
                out_channels=64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            )
            self.maxpool = nn.MaxPool2d(
                kernel_size=3,
                stride=2,
                padding=1,
            )

        # 对CIFAR等32×32小图像使用较小卷积，不立即下采样
        else:
            self.conv1 = nn.Conv2d(
                in_channels=3,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            )
            self.maxpool = nn.Identity()

        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # 四个残差阶段
        self.layer1 = self._make_layer(
            block=block,
            out_channels=64,
            blocks=layers[0],
            stride=1,
        )

        self.layer2 = self._make_layer(
            block=block,
            out_channels=128,
            blocks=layers[1],
            stride=2,
        )

        self.layer3 = self._make_layer(
            block=block,
            out_channels=256,
            blocks=layers[2],
            stride=2,
        )

        self.layer4 = self._make_layer(
            block=block,
            out_channels=512,
            blocks=layers[3],
            stride=2,
        )

        # 不管输入特征图多大，都池化为1×1
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(
            512 * block.expansion,
            num_classes,
        )

        self._initialize_weights(
            zero_init_residual=zero_init_residual
        )

    def _make_layer(
        self,
        block: Type[BasicBlock] | Type[Bottleneck],
        out_channels: int,
        blocks: int,
        stride: int,
    ) -> nn.Sequential:
        """
        构建一个残差阶段。

        一个阶段中的第一个残差块可能需要：
        1. 将特征图宽高减半；
        2. 调整输入输出通道数。

        后续残差块保持尺寸和通道不变。
        """

        output_channels = out_channels * block.expansion

        # 判断捷径分支是否需要进行尺寸或通道变换
        downsample = None

        if stride != 1 or self.in_channels != output_channels:
            downsample = nn.Sequential(
                conv1x1(
                    self.in_channels,
                    output_channels,
                    stride=stride,
                ),
                nn.BatchNorm2d(output_channels),
            )

        layers = []

        # 第一个块负责可能的下采样和通道变化
        layers.append(
            block(
                in_channels=self.in_channels,
                out_channels=out_channels,
                stride=stride,
                downsample=downsample,
            )
        )

        self.in_channels = output_channels

        # 后续块不改变尺寸
        for _ in range(1, blocks):
            layers.append(
                block(
                    in_channels=self.in_channels,
                    out_channels=out_channels,
                    stride=1,
                    downsample=None,
                )
            )

        return nn.Sequential(*layers)

    def _initialize_weights(
        self,
        zero_init_residual: bool,
    ) -> None:
        """
        参数初始化。
        """

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )

            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

        # 可选：让每个残差分支初始时更接近零映射
        if zero_init_residual:
            for module in self.modules():
                if isinstance(module, Bottleneck):
                    nn.init.constant_(module.bn3.weight, 0)

                elif isinstance(module, BasicBlock):
                    nn.init.constant_(module.bn2.weight, 0)

    def forward_features(self, x: Tensor) -> Tensor:
        """
        提取卷积特征，不经过最终全连接层。
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        x = self.forward_features(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.fc(x)

        return x