from collections import OrderedDict
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torchsummary
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

FILE_NAME = __file__.split("/")[-1][:-3]

class Model(nn.Module):
    def __init__(self, input_size=(100, 3), output_size=(3, 1), loss = None, opt = None, learning_rate = 0.001, save_path=f"./{FILE_NAME}/"):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size

        self.save_path = save_path

        self.learning_rate = learning_rate
        self.loss = loss
        self.optim = opt

        self.fc_input = nn.Linear(input_size[0] * input_size[1], 128)
        self.fc_output = nn.Linear(128, output_size[0])

        linear_list = []
        for i in range(10):
            linear_list.append((f"linear{i}", nn.Linear(128, 128)))
            linear_list.append((f"relu{i}", nn.ReLU()))

        self.seq = nn.Sequential(OrderedDict(linear_list))

        pass

    def forward(self, x):
        x = x.reshape(-1, self.input_size[0] * self.input_size[1])
        x = self.fc_input(x)
        x = F.relu(x)
        self.seq(x)
        x = self.fc_output(x)

        return x
    def compile(self):
        if self.optim is None:
            self.optim = optim.SGD(self.parameters(), self.learning_rate)
        if self.loss is None:
            self.loss = nn.MSELoss()

    def save(self, name = f"{FILE_NAME}"):
        if not os.path.isdir(self.save_path):
            os.mkdir(self.save_path)
        model_path = self.save_path + name + "_model" + ".pth"
        optim_path = self.save_path + name + "_optim" + ".pth"
        loss_path = self.save_path + name + "_loss" + ".pth"

        torch.save(self.state_dict(), model_path)
        torch.save(self.optim.state_dict(), optim_path)
        torch.save(self.loss.state_dict(), loss_path)


if __name__ == "__main__":
    model = Model()
    model.compile()
    model.save()
    # torch.save(model.state_dict(), "./temp.pth")
