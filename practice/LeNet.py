import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1=nn.Conv2d(
            in_channels=1,
            out_channels=6,
            kernel_size=5,
            stride=1,
            padding=0,
            bias=True
        )

        self.pool1=nn.AvgPool2d(kernel_size=2,stride=2)

        self.conv2=nn.Conv2d(
            in_channels=6,
            out_channels=16,
            kernel_size=5,
            stride=1,
            padding=0,
            bias=True
        )

        self.pool2=nn.AvgPool2d(kernel_size=2,stride=2)

        self.conv3=nn.Conv2d(
            in_channels=16,
            out_channels=120,
            kernel_size=5,
            stride=1,
            padding=0,
            bias=True
        )

        self.fc1=nn.Linear(120,84)
        self.fc2=nn.Linear(84,num_classes)


    def forward(self,x):
        # x [B,1,32,32]

        x=torch.tanh(self.conv1(x))
        x=self.pool1(x)

        x=torch.tanh(self.conv2(x))
        x=self.pool2(x)

        x=torch.tanh(self.conv3(x))

        x=torch.flatten(x,start_dim=1) #从第一维开始展平，第零维不变

        x=torch.tanh(self.fc1(x))
        x=self.fc2(x)

        return x
    

model=LeNet(num_classes=10)
x=torch.randn(8,1,32,32)
y=model(x)
