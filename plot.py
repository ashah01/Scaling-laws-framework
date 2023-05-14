import matplotlib.pyplot as plt
import re

class DataVisualizer:
    def __init__(self, folder):
        self.config = {"batch_size": 0, "lr": 1, "width": 2, "depth": 3, "params": 4, "dropout": 5, "loss": 6}
        self.folder = folder
        self.run = []
    
    def load_data(self, dir):
        with open(f"observations/{dir}/analytics.txt", "r") as file:
            data = file.readlines()
        run = [re.findall(r"[-+]?\d*\.\d+|\d+", line) for line in data]
        for i in range(len(run)):
            for j in range(len(run[0])):
                if run[i][j].isdigit():
                    run[i][j] = int(run[i][j])
                else:
                    run[i][j] = float(run[i][j])
        
        self.run = run

    def visualize_data(self, x):
        self.load_data(self.folder)
        import IPython; IPython.embed()
        plt.plot([row[self.config[x]] for row in self.run], [row[self.config["loss"]] for row in self.run], marker="o")
        plt.xlabel(x)
        plt.ylabel("loss")
        plt.show()
    
    def visualize_size(self):
        assert type(self.folder) == list
        for folder in self.folder:
            self.load_data(folder)
            plt.plot([row[self.config["params"]] for row in self.run], [row[self.config["loss"]] for row in self.run], label=folder, marker="o")
        plt.legend()
        plt.xlabel("params")
        plt.ylabel("loss")
        plt.show()
