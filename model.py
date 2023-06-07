import torch
import torch.nn as nn



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
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1, downsample = None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU())
        self.conv2 = nn.Sequential(
                        nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),
                        nn.BatchNorm2d(out_channels))
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, num_filters, layers):
        super(ResNet, self).__init__()
        self.inplanes = num_filters
        self.conv1 = nn.Sequential(
                        nn.Conv2d(3, num_filters, kernel_size = 7, stride = 2, padding = 3),
                        nn.BatchNorm2d(num_filters),
                        nn.ReLU())
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.layer0 = self._make_layer(ResidualBlock, num_filters, layers, stride = 1)
        self.layer1 = self._make_layer(ResidualBlock, num_filters*2, layers, stride = 2)
        self.layer2 = self._make_layer(ResidualBlock, num_filters*4, layers, stride = 2)
        self.avgpool = nn.AvgPool2d(2, stride=1)
        self.fc = nn.Linear(num_filters*4, 10)
        self.apply(init_params)
        
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
    
# model = ResNet(16, 3).to(device)

if __name__ == "__main__":
    model = ResNet(16, 1)
    import IPython; IPython.embed()