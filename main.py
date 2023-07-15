"""Train CIFAR10 with PyTorch."""
import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import wandb

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
    with wandb.init(config=config, project="resnet-scaling-laws"):
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

        for epoch in range(wandb.config.epochs):
            net.train()
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
                        "train/loss": loss.item(),
                        "train/error %": 100.0 * (1 - correct / total),
                        "train/batch_idx": batch_idx,
                    }
                )

            net.eval()
            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(testloader):
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = net(inputs)

                    loss = criterion(outputs, targets)

                    _, predicted = outputs.max(1)
                    total = targets.size(0)
                    correct = predicted.eq(targets).sum().item()

                    wandb.log(
                        {
                            "test/loss": loss.item(),
                            "test/error %": 100.0 * (1 - correct / total),
                            "test/batch_idx": batch_idx,
                        }
                    )

hyperparameter_config = {
    "epochs": 50,
    "batch_size": 128,
    "lr": 0.1,
    "wd": 5e-4,
}

# 500k combos
combos_500k = [
    (41, 1), # 41
    (27, 2), # 13.5
    (22, 3), # 7.33
    (19, 4), # 4.75
    (11, 11), # 1
    (9, 14), # 0.64
    (8, 21), # 0.38
    (5, 53), # 0.09
]

combos_1m = [ # d_model / n_layer
    (8, 41), # 0.2
    (10, 27), # 0.37
    (12, 19), # 0.63
    (14, 14), # 1
    (24, 5), # 4.8
    (27, 4), # 6.75
    (31, 3), # 10.33
    (39, 2), # 19.5
    (58, 1), # 58
]

hyperparameter_config["num_params"] = 500000
for w, d in combos_500k:
    hyperparameter_config["hidden_dim"] = w
    hyperparameter_config["depth"] = d

    loop(hyperparameter_config)

hyperparameter_config["num_params"] = 1000000
for w, d in combos_1m:
    hyperparameter_config["hidden_dim"] = w
    hyperparameter_config["depth"] = d

    loop(hyperparameter_config)