import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.utils.data
from tqdm import tqdm
import wandb
import itertools
import model


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose(
    [
        transforms.RandomCrop(
            32, padding=4
        ),  # Random crop of size 32x32 with padding of 4 pixels
        transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
        transforms.ToTensor(),  # Convert the image to a tensor
    ]
)

trainset = torchvision.datasets.CIFAR10(
    root="./data/CIFAR10", train=True, download=True, transform=transform
)

testset = torchvision.datasets.CIFAR10(
    root="./data/CIFAR10", train=False, download=True, transform=transform
)


def train(**kwargs):
    wandb.init(
        project="resnet-scaling-laws",
        config=kwargs, 
        reinit=True, 
    ) 

    torch.manual_seed(kwargs["seed"])

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=kwargs["batch_size"], shuffle=True
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=kwargs["batch_size"], shuffle=False
    )

    net = getattr(model, kwargs["name"])(kwargs["hidden_dim"], kwargs["depth"])
    net = net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        net.parameters(), lr=kwargs["lr"], weight_decay=kwargs["wd"]
    )
    scheduler = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1,
        end_factor=0,
        total_iters=len(trainloader) * kwargs["epochs"],
    )
    for _ in range(kwargs["epochs"]):
        for data in tqdm(trainloader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            wandb.log({"train/loss": loss.item()})
            loss.backward()
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

        wandb.log({"test/error_rate": (running_sum / len(testset))})

    wandb.finish()

def recursive_call(**args):
    assert type(args["name"]) != list
    assert type(args["folder"]) != list
    search_space = dict(filter(lambda x: type(x[1]) == list, args.items()))
    constant = dict(filter(lambda x: type(x[1]) != list, args.items()))
    call_combinations(search_space, constant, train)


def call_combinations(dictionary, constant, function):
    keys = dictionary.keys()
    values = dictionary.values()
    combinations = list(itertools.product(*values))

    for combo in combinations:
        function_args = dict(zip(keys, combo))
        function(**constant, **function_args)



# polymorphism implemented through arrays
#recursive_call(
#    name="ResNet",
#    epochs=50,
#    batch_size=128,
#    lr=0.01,
#    wd=5e-4,
#    hidden_dim=16,
#    depth=[2, 3, 5, 7, 9],
#    dropout=0,
#    seed=[0, 1],
#    folder="wd_sweep",
#)
train(name="ResNet", epochs=50, batch_size=128, lr=0.01, wd=5e-4, hidden_dim=16, depth=9, dropout=0, seed=1, folder="wd_sweep")
