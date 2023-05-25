import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms


"""

import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  # Random crop of size 32x32 with padding of 4 pixels
    transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
    transforms.ToTensor()  # Convert the image to a tensor
])

trainset = torchvision.datasets.CIFAR10(root="./data/CIFAR10", train=True, download=True, transform=transform)

testset = torchvision.datasets.CIFAR10(root="./data/CIFAR10", train=False, download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=32, shuffle=True
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=32, shuffle=False
)

sample = next(iter(trainloader))[0]

"""

def init_params(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

def init_cnn(module):
    """Initialize weights for CNNs."""
    if type(module) == nn.Linear or type(module) == nn.Conv2d:
        nn.init.xavier_uniform_(module.weight)

class Residual(nn.Module):
    """The Residual block of ResNet models."""
    def __init__(self, num_channels, in_features, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_features, num_channels, kernel_size=3, padding=1,
                                   stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_features, num_channels, kernel_size=1,
                                       stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)
    
class ResNet(nn.Module):

    def __init__(self, hidden_dim, depth):
        super(ResNet, self).__init__()
        self.net = nn.ModuleList([self.b1(hidden_dim)])
        for i in range(depth):
            self.net.append(self.block(2, hidden_dim, first_block=(i==0)))
            hidden_dim *= 2
        self.net.append(nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            nn.Linear(hidden_dim//2, 10))) # [num_features, num_classes]
        self.net.apply(init_cnn)

    def b1(self, hidden_dim):
        return nn.Sequential(
            nn.Conv2d(3, hidden_dim, kernel_size=3, stride=2, padding=3),
            nn.BatchNorm2d(hidden_dim), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

    def block(self, num_residuals, num_channels, first_block=False):
        blk = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.append(Residual(num_channels, num_channels//2, use_1x1conv=True, strides=2))
            else:
                blk.append(Residual(num_channels, num_channels))
        return nn.Sequential(*blk)

    def forward(self, x):
        for layer in self.net:
            x = layer(x)
        return x

class TransMLP(nn.Module):
    def __init__(self, width, depth):
        super().__init__()
        self.first_layer = nn.Sequential(
            nn.Flatten(), nn.Linear(28*28, width), nn.ReLU()
        )
        self.middle_layers = nn.ModuleList()
        for _ in range(depth - 1):
            self.middle_layers.append(
                nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.ReLU())
            )

        self.output_layer = nn.Linear(width, 10)
        self.apply(init_params)

    def forward(self, x):
        x = self.first_layer(x)
        for layer in self.middle_layers:
            x = x + layer(x)
        x = self.output_layer(x)
        return x