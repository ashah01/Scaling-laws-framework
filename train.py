import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from tqdm import tqdm
import pickle
import os
import time
import itertools
import model
from plot import DataVisualizer
from operator import itemgetter

torch.manual_seed(1)

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
    subdir = os.path.join(f"./observations/{kwargs['name']}", kwargs["folder"])

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
    train_scores = []
    avg_test_losses = []
    time_start = time.time()
    for epoch in range(kwargs["epochs"]):
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
            scheduler.step()

        test_scores = []
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images = images.to(device)
                labels = labels.to(device)
                outputs = net(images)
                l_test = criterion(outputs, labels)
                test_scores.append(l_test.item())

        avg_test_loss = sum(test_scores) / len(test_scores)

        avg_test_losses.append(avg_test_loss)
        print(avg_test_loss)

    time_end = time.time()
    running_sum = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = torch.nn.functional.log_softmax(net(images), dim=1)
            running_sum += (outputs.argmax(dim=1) != labels).sum().item()

    print("Error %: ", (running_sum / len(testset)))

    if kwargs["save"]:
        identity_ = f"b{kwargs['batch_size']}dr{kwargs['dropout']}lr{kwargs['lr']}d{kwargs['depth']}w{kwargs['hidden_dim']}"
        with open(
            f"{subdir}/train_scores_{identity_}",
            "wb",
        ) as f:
            pickle.dump(train_scores, f)
            f.close()
        with open(
            f"{subdir}/test_scores_{identity_}",
            "wb",
        ) as f:
            pickle.dump(test_scores, f)
            f.close()
    if kwargs["log"]:
        with open(f"{subdir}/analytics.txt", "a") as f:
            f.write(
                f"batch size: {kwargs['batch_size']}, lr: {kwargs['lr']}, wd: {kwargs['wd']}, hidden dim: {kwargs['hidden_dim']}, depth: {kwargs['depth']}, params: {sum([p.numel() for p in net.parameters()])}, dropout: {kwargs['dropout']}, loss: {min(avg_test_losses)}, error %: {running_sum / len(testset)}, time: {time_end - time_start}, epochs: {kwargs['epochs']}\n"
            )
            f.close()


def recursive_call(**args):
    assert type(args["name"]) != list
    assert type(args["folder"]) != list
    subdir = os.path.join(f"./observations/{args['name']}", args["folder"])
    if not os.path.exists(subdir):
        os.makedirs(subdir)
        open(subdir + "/analytics.txt", "a").close()
    search_space = dict(filter(lambda x: type(x[1]) == list, args.items()))
    constant = dict(filter(lambda x: type(x[1]) != list, args.items()))
    import IPython; IPython.embed()
    if search_space:
        call_combinations(search_space, constant, train)
    else:
        train(**constant)


def call_combinations(dictionary, constant, function):
    keys = dictionary.keys()
    values = dictionary.values()
    combinations = list(itertools.product(*values))

    combinations = prune_combinations(
        combinations, f"{constant['name']}/{constant['folder']}", keys
    )
    for combo in combinations:
        function_args = dict(zip(keys, combo))
        function(**constant, **function_args)


def prune_combinations(combos, dir, k):
    dv = DataVisualizer(dir)
    dv.load_data(lambda x: x)
    if not dv.run:
        return combos
    indices = itemgetter(*k)(dv.config)
    final_run = itemgetter(*indices)(dv.run[-1])
    pruned = delete_items_preceding(combos, final_run)

    return pruned


def delete_items_preceding(lst, value):
    if value in lst:
        index = lst.index(value)
        del lst[: index + 1]
    return lst


# polymorphism implemented through arrays
recursive_call(
    name="ResNet",
    epochs=50,
    batch_size=128,
    lr=0.01,
    wd=1e-4,
    hidden_dim=16,
    depth=2,
    dropout=0,
    save=False,
    log=False,
    folder="wd_and_epochs",
)
