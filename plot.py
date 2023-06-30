import matplotlib.pyplot as plt
import re
import torch
import pickle
import math

class DataVisualizer:
    def __init__(self, folder):
        self.config = {"batch_size": 0, "lr": 1, "wd": 2, "width": 3, "depth": 4, "params": 5, "dropout": 6, "loss": 7, "error": 8, "time": 9, "epochs": 10}
        self.folder = folder
        self.run = []
    
    def load_data(self, filter_):
        with open(f"observations/{self.folder}/analytics.txt", "r") as file:
            data = file.readlines()
        run = [re.findall(r"[-+]?\d*\.\d+|\d+", line) for line in data]
        for i in range(len(run)):
            for j in range(len(run[0])):
                if run[i][j].isdigit():
                    run[i][j] = int(run[i][j])
                else:
                    run[i][j] = float(run[i][j])
        
        
        run = list(filter(filter_, run))
        
        self.run = run

    def visualize_data(self, x, y, filter_ = lambda x: x):
        self.load_data(filter_)
        ranges = count_occurrences([col[self.config[y]] for col in self.run])
        clumps = clump_subarrays(self.run, ranges)
        for v in clumps:
            plt.plot([i[self.config[x]] for i in v], [a[self.config['loss']] for a in v], marker="o", label=f"{y}: {v[0][self.config[y]]}")
        plt.legend()
        plt.xlabel(x)
        plt.ylabel("loss")
        if x == "lr":
            plt.xscale("log")
        plt.show()
    
    def visualize_size(self, filter_ = lambda x: x):
        self.load_data(filter_)
        plt.plot([row[self.config["params"]] for row in self.run], [row[self.config["loss"]] for row in self.run], label=self.folder, marker="o")
        plt.legend()
        plt.xlabel("params")
        plt.ylabel("loss")
        plt.show()

    
    def visualize_performance(self, run: dict):
        with open(f"observations/{self.folder}/train_scores_b{run['batch size']}dr{run['dropout']}lr{run['lr']}d{run['depth']}w{run['width']}", "rb") as f:
            train_scores = torch.tensor(pickle.load(f))
            f.close()

        with open(f"observations/{self.folder}/test_scores_b{run['batch size']}dr{run['dropout']}lr{run['lr']}d{run['depth']}w{run['width']}", "rb") as f:
            test_scores = torch.tensor(pickle.load(f))
            f.close()

        train_epochs = torch.split(train_scores, math.ceil(50000/run["batch size"]))
        test_epochs = torch.split(test_scores, math.ceil(10000/run["batch size"]))

        plt.plot([x.mean().item() for x in train_epochs], label="train", marker="o")
        plt.plot([x.mean().item() for x in test_epochs], label="test", marker="o")
        plt.legend()
        plt.show()

        best_train = min([x.mean().item() for x in train_epochs][-5:])
        best_test = min([x.mean().item() for x in test_epochs][-5:])
        print("Train-test loss difference: ",  abs(best_train - best_test))
        print("Best train loss: ", best_train)
        print("Best test loss: ", best_test)

def count_occurrences(sorted_list):
    counts = []
    current_count = 1

    # Iterate over the sorted list starting from the second element
    for i in range(1, len(sorted_list)):
        if sorted_list[i] == sorted_list[i - 1]:
            # If the current element is the same as the previous one, increment the count
            current_count += 1
        else:
            # If the current element is different from the previous one, add the count to the list
            counts.append(current_count)
            current_count = 1

    # Add the count of the last element to the list
    counts.append(current_count)

    return counts

def clump_subarrays(sorted_list, counts):
    subarrays = []

    start_index = 0
    for count in counts:
        subarray = sorted_list[start_index:start_index + count]
        subarrays.append(subarray)
        start_index += count

    return subarrays

