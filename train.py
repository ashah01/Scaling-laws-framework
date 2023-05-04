import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--hidden_dim", type=int, default=32)
parser.add_argument("--depth", type=int, default=2)
parser.add_argument("--dropout1", type=float, default=0.25)
parser.add_argument("--dropout2", type=float, default=0.5)
args = parser.parse_args()


transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3801,))]
)

trainset = torchvision.datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
testset = torchvision.datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False)

def init_params(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class Net(nn.Module):
    def __init__(self, hidden_dim, depth, d1, d2):
        super(Net, self).__init__()
        assert depth >= 2
        modules = [
            nn.Conv2d(1, hidden_dim, 3, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim * 2, 3, 1),
            nn.ReLU(),
        ]
        for _ in range(depth - 2):
            modules.append(nn.Conv2d(hidden_dim * 2, hidden_dim * 2, 3, 1))
            modules.append(nn.ReLU())
        self.convolutions = nn.Sequential(*modules)
        self.dropout1 = nn.Dropout(d1)
        self.dropout2 = nn.Dropout(d2)
        self.fc1 = nn.Linear(
            ((28 - (depth * 2)) // 2) ** 2 * hidden_dim * 2, hidden_dim * 4
        )
        self.fc2 = nn.Linear(hidden_dim * 4, 10)
        self.apply(init_params)

    def forward(self, x):
        x = self.convolutions(x)
        x = F.max_pool2d(x, 2)  # image dim reduced in half
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


net = Net(args.hidden_dim, args.depth, args.dropout1, args.dropout2)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=args.lr)

prev_avg_loss = float("inf")
avg_test_loss = float("inf") # scoping
while True:
    train_scores = []
    for data in tqdm(trainloader):
        inputs, labels = data

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        train_scores.append(loss.item())
        loss.backward()
        optimizer.step()
    
    test_scores = []
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            l_test = criterion(outputs, labels)
            test_scores.append(l_test.item())
    
    avg_test_loss = sum(test_scores) / len(test_scores)
    print(avg_test_loss)

    if avg_test_loss < prev_avg_loss:
        prev_avg_loss = avg_test_loss
        continue
    else:
        import IPython; IPython.embed()
        # plt.plot(train_scores[-313:], label="train")
        # plt.plot(test_scores, label="test")
        # plt.legend()
        # plt.savefig("capacity.png")
        break


with open("analytics.txt", "a") as f:
    f.write(f"batch size: {args.batch_size}, lr: {args.lr}, hidden dim: {args.hidden_dim}, depth: {args.depth}, params: {sum([p.numel() for p in net.parameters()])}, dropout1: {args.dropout1}, dropout2: {args.dropout2}, loss: {prev_avg_loss}\n")
    f.close()