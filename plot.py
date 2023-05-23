import matplotlib.pyplot as plt
import re
import torch
import pickle
import math

class DataVisualizer:
    def __init__(self, folder):
        self.config = {"batch_size": 0, "lr": 1, "width": 2, "depth": 3, "params": 4, "dropout": 5, "loss": 6}
        self.folder = folder
        self.run = []
    
    def load_data(self, dir, filter_):
        with open(f"observations/{dir}/analytics.txt", "r") as file:
            data = file.readlines()
        run = [re.findall(r"[-+]?\d*\.\d+|\d+", line) for line in data]
        for i in range(len(run)):
            for j in range(len(run[0])):
                if run[i][j].isdigit():
                    run[i][j] = int(run[i][j])
                else:
                    run[i][j] = float(run[i][j])
        
        if filter_:
            for key in filter_:
                run = [row for row in run if row[self.config[key]] == filter_[key]]
        
        self.run = run

    def visualize_data(self, x, filter_: dict = {}):
        for folder in self.folder:
            self.load_data(folder, filter_)
            plt.plot([row[self.config[x]] for row in self.run], [row[self.config["loss"]] for row in self.run], marker="o", label=folder)
        plt.legend()
        plt.xlabel(x)
        plt.ylabel("loss")
        plt.show()
    
    def visualize_size(self, filter_: dict = {}):
        assert type(self.folder) == list
        for folder in self.folder:
            self.load_data(folder, filter_)
            plt.plot([row[self.config["params"]] for row in self.run], [row[self.config["loss"]] for row in self.run], label=folder, marker="o")
        plt.legend()
        plt.xlabel("params")
        plt.ylabel("loss")
        plt.show()

    
    def visualize_capacity(self, run: dict):
        for folder in self.folder:
            with open(f"observations/{folder}/train_scores_b{run['batch size']}dr{run['dropout']}lr{run['lr']}d{run['depth']}w{run['width']}", "rb") as f:
                train_scores = torch.tensor(pickle.load(f))
                f.close()

            with open(f"observations/{folder}/test_scores_b{run['batch size']}dr{run['dropout']}lr{run['lr']}d{run['depth']}w{run['width']}", "rb") as f:
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
            print("Train-test loss difference: ",  best_train - best_test)
            print("Best train loss: ", best_train)
            print("Best test loss: ", best_test)