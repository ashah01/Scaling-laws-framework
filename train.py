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
import model

torch.manual_seed(3407)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3801,))]
)

trainset = torchvision.datasets.FashionMNIST(
    root="./data", train=True, download=True, transform=transform
)
testset = torchvision.datasets.FashionMNIST(
    root="./data", train=False, download=True, transform=transform
)


def train(args):
    subdir = os.path.join(f"./observations/{args.name}", args.folder)
    if not os.path.exists(subdir):
        os.makedirs(subdir)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False
    )
    net = getattr(model, args.name)(args.hidden_dim, args.depth)
    net = net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=args.lr)

    train_scores = []
    test_scores = []
    avg_test_losses = []
    num_test_batches = math.ceil(10000 / args.batch_size)
    p = 4
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
        else:
            p = 4

        if p == 0:
            break

    if args.save:
        with open(
            f"observations/{args.name}/{args.folder}/train_scores_b{args.batch_size}dr{args.dropout}lr{args.lr}d{args.depth}w{args.hidden_dim}",
            "wb",
        ) as f:
            pickle.dump(train_scores, f)
            f.close()
        with open(
            f"observations/{args.name}/{args.folder}/test_scores_b{args.batch_size}dr{args.dropout}lr{args.lr}d{args.depth}w{args.hidden_dim}",
            "wb",
        ) as f:
            pickle.dump(test_scores, f)
            f.close()
    if args.log:
        with open(f"observations/{args.name}/{args.folder}/analytics.txt", "a") as f:
            f.write(
                f"batch size: {args.batch_size}, lr: {args.lr}, hidden dim: {args.hidden_dim}, depth: {args.depth}, params: {sum([p.numel() for p in net.parameters()])}, dropout: {args.dropout}, loss: {min(avg_test_losses)}\n"
            )
            f.close()


for dp in [2, 3]:
    for l in [1e-4]:
        for dr in [0]:
            train(
                Namespace(
                    name="TransMLP",
                    batch_size=32,
                    lr=l,
                    hidden_dim=64,
                    depth=dp,
                    dropout=dr,
                    save=True,
                    log=True,
                    folder="diagnose_depthoptima_fashionmnist",
                )
            )

