import torch
import torch.nn as nn

class LeNet1(nn.Module):
    '''
    LeNet1 model (same with ATS and RTS)
    '''
    def __init__(self):
        super(LeNet1, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=5, stride=(1, 1), padding=2)
        self.conv2 = nn.Conv2d(4, 12, kernel_size=5, stride=(1, 1), padding=2)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(588, 10)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.pool2(self.relu(self.conv2(x)))
        # x = x.view(batch_size, -1)
        x = x.permute(0, 2, 3, 1)  # keep same flatten operate with keras model
        x = x.reshape(batch_size, -1)
        x = self.softmax(self.fc1(x))

        return x