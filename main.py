"""Train CIFAR10 with PyTorch."""
import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import wandb

import os

from models.resnet import ResNet

torch.manual_seed(0)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Data
transform_train = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

transform_test = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

trainset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform_train
)


testset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform_test
)


def loop(config=None):
    with wandb.init(config=config, project="resnet-scaling-laws", resume=True):
        wandb.define_metric("train/step")
        wandb.define_metric("train/*", step_metric="train/step")
        wandb.define_metric("test/step")
        wandb.define_metric("test/*", step_metric="test/step")
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=wandb.config.batch_size, shuffle=True
        )
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=wandb.config.batch_size, shuffle=False
        )
        net = ResNet(wandb.config.hidden_dim, wandb.config.depth)
        net = net.to(device)
        optimizer = optim.SGD(
            net.parameters(),
            lr=wandb.config.lr,
            momentum=0.9,
            weight_decay=wandb.config.wd,
        )
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=wandb.config.epochs * len(trainloader)
        )

        epoch_subtract = 0
        if os.path.exists("./last_checkpoint.pt"):
            checkpoint = torch.load("./last_checkpoint.pt")
            net.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            epoch_subtract = checkpoint["epoch"] - 1

        for epoch in range(wandb.config.epochs - epoch_subtract):
            net.train()

            if (epoch + 1) % 5 == 0:
                # checkpoint
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": net.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                    },
                    "./last_checkpoint.pt",
                )

            for batch_idx, (inputs, targets) in enumerate(trainloader):
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = net(inputs)

                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                scheduler.step()

                _, predicted = outputs.max(1)
                total = targets.size(0)
                correct = predicted.eq(targets).sum().item()

                wandb.log(
                    {
                        "train/step": epoch * len(trainloader) + batch_idx + 1,
                        "train/loss": loss.item(),
                        "train/error %": 100.0 * (1 - correct / total),
                    }
                )

            net.eval()
            with torch.no_grad():
                test_losses = []
                test_errors = []
                for batch_idx, (inputs, targets) in enumerate(testloader):
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = net(inputs)

                    loss = criterion(outputs, targets)

                    _, predicted = outputs.max(1)
                    total = targets.size(0)
                    correct = predicted.eq(targets).sum().item()

                    test_losses.append(loss.item())
                    test_errors.append(100.0 * (1 - correct / total))
                wandb.log({
                    "test/step": epoch * len(testloader) + batch_idx + 1,
                    "test/loss": sum(test_losses) / len(test_losses),
                    "test/error %": sum(test_errors) / len(test_errors),
                    "test/error std": torch.tensor(test_errors).std().item(),
                    "test/error min": min(test_errors),
                    "test/error max": max(test_errors),
                })

        os.remove("./last_checkpoint.pt")


hyperparameter_config = {
    "epochs": 50,
    "batch_size": 128,
    "lr": 0.1,
    "wd": 5e-4,
    "hidden_dim": 24,
    "depth": 5
}

# for w,d in [(35, 7), (24, 5)]:
#     hyperparameter_config['hidden_dim'] = w
#     hyperparameter_config['depth'] = d
loop(hyperparameter_config)