import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from tqdm import tqdm
import math
import pickle
from argparse import Namespace
import os

torch.manual_seed(3407)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3801,))]
)

trainset = torchvision.datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
testset = torchvision.datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)


def init_params(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class MLP(nn.Module):
    def __init__(self, embedding_dim, num_layers, dropout_rate=0.1):
        super().__init__()

        # Input layer
        self.layers = nn.ModuleList([nn.Flatten(), nn.Linear(784, embedding_dim)])

        # Hidden layers
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(embedding_dim, embedding_dim))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(p=dropout_rate))

        self.layers.append(nn.Linear(embedding_dim, 10))

        self.apply(init_params)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def train(args):
    subdir = os.path.join("./observations", args.folder)
    if not os.path.exists(subdir):
        os.makedirs(subdir)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False
    )

    net = MLP(args.hidden_dim, args.depth, args.dropout)
    net = net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=args.lr)

    train_scores = []
    test_scores = []
    avg_test_losses = []
    num_test_batches = math.ceil(10000 / args.batch_size)

    p = 5

    while True:
        for data in tqdm(trainloader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            train_scores.append(loss.item())
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images = images.to(device)
                labels = labels.to(device)
                outputs = net(images)
                l_test = criterion(outputs, labels)
                test_scores.append(l_test.item())

        avg_test_loss = sum(test_scores[-num_test_batches:]) / len(
            test_scores[-num_test_batches:]
        )

        avg_test_losses.append(avg_test_loss)
        print(avg_test_loss)

        if avg_test_loss > min(avg_test_losses):
            p -= 1
        
        if p == 0:
            break

    if args.save:
        with open(
            f"observations/{args.folder}/train_scores_b{args.batch_size}dr{args.dropout}lr{args.lr}d{args.depth}w{args.hidden_dim}",
            "wb",
        ) as f:
            pickle.dump(train_scores, f)
            f.close()
        with open(
            f"observations/{args.folder}/test_scores_b{args.batch_size}dr{args.dropout}lr{args.lr}d{args.depth}w{args.hidden_dim}",
            "wb",
        ) as f:
            pickle.dump(test_scores, f)
            f.close()
    if args.log:
        with open(f"observations/{args.folder}/analytics.txt", "a") as f:
            f.write(
                f"batch size: {args.batch_size}, lr: {args.lr}, hidden dim: {args.hidden_dim}, depth: {args.depth}, params: {sum([p.numel() for p in net.parameters()])}, dropout: {args.dropout}, loss: {min(avg_test_losses)}\n"
            )
            f.close()


# TODO: build smarter HP configuration generation

for depth in [2]:
    for lr in [1e-4, 3e-4]:
        train(
            Namespace(
                batch_size=32,
                lr=lr,
                hidden_dim=32,
                depth=depth,
                dropout=0,
                save=False,
                log=True,
                folder="smallerlr_thindeep",
            )
        )
