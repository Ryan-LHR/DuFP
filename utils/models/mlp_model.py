import torch
import torch.nn as nn

class MLP_Model(nn.Module):
    '''
    MLP model with custom input dimensions
    '''

    def __init__(self, input_dimensions):
        self.fc1 = nn.Linear(input_dimensions, 128)  # 第一个隐藏层
        self.fc2 = nn.Linear(128, 128)  # 第二个隐藏层
        self.fc3 = nn.Linear(128, input_dimensions)  # 输出层
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x