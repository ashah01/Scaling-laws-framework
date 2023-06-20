import argparse
import os
import re
import pickle

config = {"batch_size": 0, "lr": 1, "width": 2, "depth": 3, "params": 4, "dropout": 5, "loss": 6, "error": 7, "time": 8, "epochs": 9}

parser = argparse.ArgumentParser(description='Aggregate results from multiple runs')
parser.add_argument('--folder', type=str, required=True, help="The name of the destination folder")
parser.add_argument('--sort', type=str, required=True, help="The key by which to sort the values")
parser.add_argument('--sort2', type=str, required=True, help="The key by which to sort the values")
parser.add_argument('--names', nargs='+', type=str, required=True, help="The names of the runs")


args = parser.parse_args()
# Get values
#   Load all entries as vectors where each column represents a property
entries = []
for folder in args.names:
    subdir = os.path.join("observations/ResNet", folder, "analytics.txt")
    if not os.path.exists(subdir):
        raise ValueError(f"File {subdir} does not exist")
    with open(subdir, "r") as f:
        entries.extend(f.readlines())
        f.close()

run = [re.findall(r"[-+]?\d*\.\d+|\d+", line) for line in entries]
for i in range(len(run)):
    for j in range(len(run[0])):
        if run[i][j].isdigit():
            run[i][j] = int(run[i][j])
        else:
            run[i][j] = float(run[i][j])

# Sort them
import IPython; IPython.embed()
entries = sorted(run, key=lambda x: (x[config[args.sort]], x[config[args.sort2]]))

# Save to new folder
subdir = os.path.join(f"./observations/ResNet", args.folder, "analytics.txt")
if not os.path.exists(subdir):
    os.makedirs("./observations/ResNet/" + args.folder)
    open(subdir, 'w').close()

#   Decompress vector into entries
with open(subdir, 'w') as file:
    for entry in entries:
        file.write(f"batch size: {entry[0]}, lr: {entry[1]}, hidden dim: {entry[2]}, depth: {entry[3]}, params: {entry[4]}, dropout: {entry[5]}, loss: {entry[6]}, error %: {entry[7]}, time: {entry[8]}, epochs: {entry[9]}\n")