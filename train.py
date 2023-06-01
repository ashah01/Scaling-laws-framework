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
import time
import model

torch.manual_seed(3407)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  # Random crop of size 32x32 with padding of 4 pixels
    transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
    transforms.ToTensor()  # Convert the image to a tensor
])

trainset = torchvision.datasets.CIFAR10(root="./data/CIFAR10", train=True, download=True, transform=transform)

testset = torchvision.datasets.CIFAR10(root="./data/CIFAR10", train=False, download=True, transform=transform)

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
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=0.0001)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=4, min_lr=0.001)
    train_scores = []
    test_scores = []
    avg_test_losses = []
    num_test_batches = math.ceil(10000 / args.batch_size)
    p = 4
    time_start = time.time()
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

        # scheduler.step(avg_test_loss)

        avg_test_losses.append(avg_test_loss)
        print(avg_test_loss)

        if avg_test_loss > min(avg_test_losses): # (scheduler.optimizer.param_groups[0]['lr'] - 0.001) < 0.009
            p -= 1
        else:
            p = 4

        if p == 0:
            break
    time_end = time.time()
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
                f"batch size: {args.batch_size}, lr: {args.lr}, hidden dim: {args.hidden_dim}, depth: {args.depth}, params: {sum([p.numel() for p in net.parameters()])}, dropout: {args.dropout}, loss: {min(avg_test_losses)}, time: {time_end - time_start}\n"
            )
            f.close()
for hd in [8]:
    for lr in [0.1]:
        train(
            Namespace(
                name="ResNet",
                batch_size=32,
                lr=lr,
                hidden_dim=hd,
                depth=5,
                dropout=0,
                save=False,
                log=True,
                folder="lrtime_acrosswidths",
            )
        )