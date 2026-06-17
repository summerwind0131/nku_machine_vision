import torch
import torch.nn as nn
import torchvision.models as models

class AlexNet(nn.Module):
    def __init__(self,num_classes=1000):
        super(AlexNet,self).__init__()

        self.features=nn.Sequential(
            nn.Conv2d(3,96,kernel_size=11,stride=4),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5),
            nn.MaxPool2d(kernel_size=3,stride=2),

            nn.Conv2d(96,256,kernel_size=5,padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5),
            nn.MaxPool2d(kernel_size=3,stride=2),

            nn.Conv2d(256,384,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384,384,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.classfier=nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256*6*6,4096),
            nn.ReLU(inplace=True),

            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),

            nn.Linear(4096, num_classes)
        )

    def forward(self,x):
        x=self.features(x)
        x=torch.flatten(x,1)
        x=self.classfier(x)
        return x
    

model = AlexNet(num_classes=1000)
x = torch.randn(1, 3, 227, 227)
y = model(x)
print(y.shape)

model=models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
model.classifier[6]=nn.Linear(4096,5)