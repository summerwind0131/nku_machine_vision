import torch.nn as nn
from torchvision import models

def build_resnet18(num_classes=6,pretrained=True):
    if pretrained:
        weights=models.ResNet18_Weights.DEFAULT
    else:
        weights=None
    model=models.resnet18(weights=weights)
    in_features=model.fc.in_features
    model.fc=nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features,256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(256,num_classes)
    )
    return model