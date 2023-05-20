import matplotlib.pyplot as plt
import re

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
