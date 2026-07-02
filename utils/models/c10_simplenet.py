import torch
import torch.nn as nn

class C10_SimpleNet(torch.nn.Module):
    '''
    The original model structure of simple_cm
    From Arachne
    '''
    def __init__(self):
        super(C10_SimpleNet, self).__init__()
        # self.zero_pad = nn.ConstantPad2d(padding=(1, 1, 1, 1), value=0)  # 各一列的zero padding
        self.conv_v1 = nn.Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=1, padding_mode='zeros')
        self.bn = nn.BatchNorm2d(num_features=16)
        self.relu = nn.ReLU()
        self.fc_v1 = nn.Linear(1024, 512)
        self.fc_v2 = nn.Linear(512, 10)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(4, 4), stride=(4, 4), ceil_mode=True)

    def forward(self, x):
        batch_size = x.size(0)
        # x = x.view(batch_size, -1)
        x = self.conv_v1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.maxpool1(x)
        # x = x.reshape(1024)  # reshape
        x = x.view(batch_size, -1)
        x = self.fc_v1(x)
        x = self.relu(x)
        x = self.fc_v2(x)
        return x
