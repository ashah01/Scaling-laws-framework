import torch
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.utils.data
from tqdm import tqdm
import wandb
from model import ResNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose(
    [
        transforms.RandomCrop(
            32, padding=4
        ),  # Random crop of size 32x32 with padding of 4 pixels
        transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
        transforms.ToTensor(),  # Convert the image to a tensor
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

trainset = torchvision.datasets.CIFAR10(
    root="./data/CIFAR10", train=True, download=True, transform=transform
)

testset = torchvision.datasets.CIFAR10(
    root="./data/CIFAR10", train=False, download=True, transform=transform
)


def main(config=None):
    with wandb.init(config=config):
        torch.manual_seed(wandb.config.seed)

        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=wandb.config.batch_size, shuffle=True
        )
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=wandb.config.batch_size, shuffle=False
        )

        net = ResNet(wandb.config.hidden_dim, wandb.config.depth)
        net = net.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(
            net.parameters(), lr=wandb.config.lr, weight_decay=wandb.config.wd
        )
        scheduler = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1,
            end_factor=0,
            total_iters=len(trainloader) * wandb.config.epochs,
        )
        for epoch in range(wandb.config.epochs):
            for data in tqdm(trainloader):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                outputs = net(inputs)
                loss = criterion(outputs, labels)
                wandb.log({"train/loss": loss.item()})
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

            running_sum = 0
            with torch.no_grad():
                for data in testloader:
                    images, labels = data
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = net(images)
                    l_test = criterion(outputs, labels)
                    wandb.log({"test/loss": l_test.item()})
                    pred = torch.nn.functional.log_softmax(outputs, dim=1)
                    running_sum += (pred.argmax(dim=1) != labels).sum().item()

            wandb.log({"test/error_rate": (running_sum / len(testset)), "epoch": epoch})


# Define the hyperparameter lists
hyperparameter_lists = {
    "epochs": {"value": 50},
    "batch_size": {"value": 128},
    "lr": {"value": 0.01},
    "wd": {"value": 5e-4},
    "hidden_dim": {"value": 16},
    "dropout": {"value": 0},
    "depth": {"values": [3, 6, 9, 12, 15]},
    "seed": {"value": 10},
}

sweep_configuration = {
    "name": "Spaced out depth scaling",
    "metric": {"name": "test/loss", "goal": "minimize"},
    "method": "grid",
    "parameters": hyperparameter_lists,
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project="resnet-scaling-laws")

# run the sweep
wandb.agent(sweep_id, function=main)
