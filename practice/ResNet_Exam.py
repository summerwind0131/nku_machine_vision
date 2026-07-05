import torch
import torch.nn as nn
from typing import Type, List, Optional

class BasicBlock(nn.Module):
    expansion=1

    def __init__(
            self,
            in_channels:int,
            channels:int,
            stride:int=1,
            downsample:Optional[nn.Module]=None
    ):
        super.__init__()

        self.conv1=nn.Conv2d(
            in_channels=in_channels,
            out_channels=channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=True
        )
        self.bn1=nn.BatchNorm2d(channels)
        self.relu=nn.ReLU(inplace=True)

        self.conv2=nn.Conv2d(
            in_channels=in_channels,
            out_channels=channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True
        )
        self.bn2 = nn.BatchNorm2d(channels)

        # shortcut 分支
        # 输入输出尺寸不同时，使用 1×1 卷积进行变换
        self.downsample = downsample

    def forward(self,x:torch.Tensor)->torch.Tensor:
        identity=x

        out=self.conv1(x)
        out=self.bns(out)
        out=self.relu(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out=out+identity
        out=self.relu(x)

        return out
    


class Bottleneck(nn.Module):
    # Bottleneck 最终输出通道数为基础通道数的 4 倍
    expansion = 4

    def __init__(
        self,
        in_channels: int,
        channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None
    ):
        super().__init__()

        # 第一个 1×1 卷积：降低通道数
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=channels,
            kernel_size=1,
            stride=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(channels)

        # 3×3 卷积：提取空间特征
        # stride=2 时在这里降低特征图尺寸
        self.conv2 = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(channels)

        # 最后一个 1×1 卷积：恢复/扩张通道数
        self.conv3 = nn.Conv2d(
            in_channels=channels,
            out_channels=channels * self.expansion,
            kernel_size=1,
            stride=1,
            bias=False
        )
        self.bn3 = nn.BatchNorm2d(channels * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        # 1×1：降维
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # 3×3：提取特征
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        # 1×1：升维
        out = self.conv3(out)
        out = self.bn3(out)

        # shortcut 尺寸不一致时进行变换
        if self.downsample is not None:
            identity = self.downsample(x)

        # 残差相加
        out = out + identity
        out = self.relu(out)

        return out
# ============================================================
# 3. 完整 ResNet 网络
# ============================================================
class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[nn.Module],
        layers: List[int],
        num_classes: int = 1000,
        in_channels: int = 3
    ):
        super().__init__()

        # 当前特征图通道数
        self.in_channels = 64

        # 输入：
        # [N, 3, 224, 224]
        #
        # 输出：
        # [N, 64, 112, 112]
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # [N, 64, 112, 112]
        # ->
        # [N, 64, 56, 56]
        self.maxpool = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=1
        )

        # conv2_x
        # 输出空间尺寸：56×56
        self.layer1 = self._make_layer(
            block=block,
            channels=64,
            blocks=layers[0],
            stride=1
        )

        # conv3_x
        # 输出空间尺寸：28×28
        self.layer2 = self._make_layer(
            block=block,
            channels=128,
            blocks=layers[1],
            stride=2
        )

        # conv4_x
        # 输出空间尺寸：14×14
        self.layer3 = self._make_layer(
            block=block,
            channels=256,
            blocks=layers[2],
            stride=2
        )

        # conv5_x
        # 输出空间尺寸：7×7
        self.layer4 = self._make_layer(
            block=block,
            channels=512,
            blocks=layers[3],
            stride=2
        )

        # 将任意大小的特征图变成 1×1
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # BasicBlock：512 × 1
        # Bottleneck：512 × 4 = 2048
        self.fc = nn.Linear(
            512 * block.expansion,
            num_classes
        )

        self._initialize_weights()

    def _make_layer(
        self,
        block: Type[nn.Module],
        channels: int,
        blocks: int,
        stride: int
    ) -> nn.Sequential:
        """
        构造一个 stage，例如 ResNet34 的 conv3_x：

        [3×3, 128]
        [3×3, 128] × 4
        """

        downsample = None

        output_channels = channels * block.expansion

        # 当空间尺寸变化或通道数变化时，
        # shortcut 使用 1×1 卷积进行调整
        if stride != 1 or self.in_channels != output_channels:
            downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels=self.in_channels,
                    out_channels=output_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(output_channels)
            )

        layers = []

        # 一个 stage 的第一个 block 可能需要：
        # 1. stride=2，降低空间尺寸
        # 2. downsample，调整 shortcut
        layers.append(
            block(
                in_channels=self.in_channels,
                channels=channels,
                stride=stride,
                downsample=downsample
            )
        )

        # 更新当前通道数
        self.in_channels = output_channels

        # stage 中剩下的 block 不再改变尺寸
        for _ in range(1, blocks):
            layers.append(
                block(
                    in_channels=self.in_channels,
                    channels=channels,
                    stride=1,
                    downsample=None
                )
            )

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """初始化网络参数。"""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu"
                )

            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入：[N, 3, 224, 224]

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # [N, 64, 112, 112]
        x = self.maxpool(x)

        # [N, 64, 56, 56]，ResNet18/34
        # [N, 256, 56, 56]，ResNet50/101/152
        x = self.layer1(x)

        # [N, 128, 28, 28] 或 [N, 512, 28, 28]
        x = self.layer2(x)

        # [N, 256, 14, 14] 或 [N, 1024, 14, 14]
        x = self.layer3(x)

        # [N, 512, 7, 7] 或 [N, 2048, 7, 7]
        x = self.layer4(x)

        # [N, C, 1, 1]
        x = self.avgpool(x)

        # [N, C, 1, 1] -> [N, C]
        x = torch.flatten(x, 1)

        # [N, num_classes]
        x = self.fc(x)

        return x


# ============================================================
# 4. 不同版本的 ResNet
# ============================================================
def resnet18(num_classes: int = 1000, in_channels: int = 3) -> ResNet:
    return ResNet(
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        num_classes=num_classes,
        in_channels=in_channels
    )


def resnet34(num_classes: int = 1000, in_channels: int = 3) -> ResNet:
    return ResNet(
        block=BasicBlock,
        layers=[3, 4, 6, 3],
        num_classes=num_classes,
        in_channels=in_channels
    )


def resnet50(num_classes: int = 1000, in_channels: int = 3) -> ResNet:
    return ResNet(
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        num_classes=num_classes,
        in_channels=in_channels
    )


def resnet101(num_classes: int = 1000, in_channels: int = 3) -> ResNet:
    return ResNet(
        block=Bottleneck,
        layers=[3, 4, 23, 3],
        num_classes=num_classes,
        in_channels=in_channels
    )


def resnet152(num_classes: int = 1000, in_channels: int = 3) -> ResNet:
    return ResNet(
        block=Bottleneck,
        layers=[3, 8, 36, 3],
        num_classes=num_classes,
        in_channels=in_channels
    )


# ============================================================
# 5. 测试代码
# ============================================================
if __name__ == "__main__":
    # 假设进行 10 分类
    model = resnet18(num_classes=10)

    # batch_size=4，RGB 图片大小 224×224
    x = torch.randn(4, 3, 224, 224)

    output = model(x)

    print(model)
    print("输入形状：", x.shape)
    print("输出形状：", output.shape)