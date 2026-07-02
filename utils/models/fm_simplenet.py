import torch
import torch.nn as nn

class FM_SimpleNet(torch.nn.Module):
    '''
    The original model structure of simple_fm
    From Arachne
    '''

    def __init__(self):
        super(FM_SimpleNet, self).__init__()
        self.fc_v1 = nn.Linear(784, 100)
        self.relu = nn.ReLU()
        self.fc_v2 = nn.Linear(100, 10)


    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        x = self.fc_v1(x)
        x = self.relu(x)
        x = self.fc_v2(x)
        return x
