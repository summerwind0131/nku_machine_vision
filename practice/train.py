import torch
import torch.nn as nn
import torch.optim as optim
import torchvision 
import torchvision.transforms as transforms

# 1. 定义 LeNet 网络
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()

        # CIFAR10 输入是 3×32×32
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        # 32 - 5 + 1 = 28，所以输出 6×28×28

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # 28×28 -> 14×14

        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        # 14 - 5 + 1 = 10，所以输出 16×10×10
        # pool 后变成 16×5×5

        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)

        x = self.relu(self.conv2(x))
        x = self.pool(x)

        x = torch.flatten(x, start_dim=1)

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        return x

def main():
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("当前设备：", device)


    # transforms的操作
    # randomresizedCrop
    # randomHorizontlFlip
    # ToTensor
    # Normalize
    # Resize
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5,0.5,0.5),
            (0.5,0.5,0.5)
        )
    ])

    # 数据加载
    # dataset
    # batch_size
    # shuffle
    # num_workers Linux/Mac 2-8   Windows 0
    train_set=torchvision.datasets.CIFAR10(
        root="../data/",
        train=True,
        download=True,
        transform=transform
    )

    train_loader=torch.utils.data.DataLoader(
        train_set,
        batch_size=32,
        shuffle=True,
        num_workers=0
    )

    val_set=torchvision.datasets.CIFAR10(
        root="../data/",
        train=False,
        download=True,
        transform=transform
        )
    
    val_loader=torch.utils.data.DataLoader(
        val_set,
        batch_size=5000,
        shuffle=False,
        num_workers=0
    )

    classes = (
        "plane", "car", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    )

    net=LeNet().to(device)

    # 网络训练
    criterion=nn.CrossEntropyLoss()

    optimizer=optim.Adam(net.parameters(),lr=0.01)

    epochs=10

    for epoch in range(epochs):
        net.train()
        running_loss=0.0

        for step, data in enumerate(train_loader):
            images,labels=data
            images,labels=images.to(device),labels.to(device)
            optimizer.zero_grad()
            outputs=net(images)
            
            loss=criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if step % 500 == 499:
                print(
                    f"epoch: {epoch + 1}, step: {step + 1}, "
                    f"loss: {running_loss / 500:.3f}"
                )
                running_loss = 0.0

        # 每个 epoch 后验证一次
        net.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = net(images)
                predicted = torch.argmax(outputs, dim=1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        acc = correct / total
        print(f"epoch {epoch + 1} validation accuracy: {acc:.4f}")

    print("训练结束")

    # 保存模型
    torch.save(net.state_dict(), "LeNet_CIFAR10.pth")
    print("模型已保存为 LeNet_CIFAR10.pth")


if __name__ == "__main__":
    main()


