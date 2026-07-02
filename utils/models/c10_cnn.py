import torch.nn as nn
import torch

class C10_CNN(nn.Module):
    '''
    The original model structure of CNN1
    From Arachne and Apricot
    '''
    def __init__(self):
        super(C10_CNN, self).__init__()
        self.conv_v1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=1, padding_mode='zeros')
        self.conv_v2 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=1, padding_mode='zeros')
        self.conv_v3 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=1, padding_mode='zeros')
        self.conv_v4 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=1, padding_mode='zeros')
        self.maxpool_v1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), ceil_mode=True)
        self.fc_v1 = nn.Linear(8192, 256)
        self.fc_v2 = nn.Linear(256, 256)
        self.fc_v3 = nn.Linear(256, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        batch_size = x.size(0)
        x = self.conv_v1(x)
        x = self.relu(x)
        x = self.conv_v2(x)
        x = self.relu(x)
        x = self.maxpool_v1(x)
        x = self.conv_v3(x)
        x = self.relu(x)
        x = self.conv_v4(x)
        x = self.relu(x)
        x = self.maxpool_v1(x)

        x = x.view(batch_size, -1)
        x = self.fc_v1(x)
        x = self.relu(x)
        x = self.fc_v2(x)
        x = self.relu(x)
        x = self.fc_v3(x)
        return x

#
# class C10_CNN1_Net_withForword2_v2(torch.nn.Module):
#     '''
#     The original model structure of CNN1, added fc layers
#     From Arachne and Apricot
#         为deeplift构造的模型(relu函数分别定义)
#     '''
#     def __init__(self):
#         super(C10_CNN1_Net_withForword2_v2, self).__init__()
#         self.conv_v1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=1, padding_mode='zeros')
#         self.conv_v2 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=1, padding_mode='zeros')
#         self.conv_v3 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=1, padding_mode='zeros')
#         self.conv_v4 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=1, padding_mode='zeros')
#         self.maxpool_v1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), ceil_mode=True)
#         self.maxpool_v2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), ceil_mode=True)
#         self.fc_v1 = nn.Linear(8192, 256)
#         self.fc_v2 = nn.Linear(256, 256)
#         self.fc_v3 = nn.Linear(256, 10)
#         self.relu_v1 = nn.ReLU()
#         self.relu_v2 = nn.ReLU()
#         self.relu_v3 = nn.ReLU()
#         self.relu_v4 = nn.ReLU()
#         self.relu_v5 = nn.ReLU()
#         self.relu_v6 = nn.ReLU()
#
#         # added fc layers
#         self.fc_v4 = nn.Linear(8192, 256)
#         self.fc_v5 = nn.Linear(256, 256)
#         self.fc_v6 = nn.Linear(256, 2)
#
#
#     def forward(self, x):
#         batch_size = x.size(0)
#         x = self.conv_v1(x)
#         x = self.relu_v1(x)
#         x = self.conv_v2(x)
#         x = self.relu_v2(x)
#         x = self.maxpool_v1(x)
#         x = self.conv_v3(x)
#         x = self.relu_v3(x)
#         x = self.conv_v4(x)
#         x = self.relu_v4(x)
#         x = self.maxpool_v2(x)
#
#         x = x.view(batch_size, -1)
#         x = self.fc_v1(x)
#         x = self.relu_v5(x)
#         x = self.fc_v2(x)
#         x = self.relu_v6(x)
#         x = self.fc_v3(x)
#         return x
#
#     def forward_v2(self, x):
#         batch_size = x.size(0)
#         x = self.conv_v1(x)
#         x = self.relu_v1(x)
#         x = self.conv_v2(x)
#         x = self.relu_v1(x)
#         x = self.maxpool_v1(x)
#         x = self.conv_v3(x)
#         x = self.relu_v1(x)
#         x = self.conv_v4(x)
#         x = self.relu_v1(x)
#         x = self.maxpool_v1(x)
#
#         x = x.view(batch_size, -1)
#         x = self.fc_v4(x)
#         x = self.relu_v1(x)
#         x = self.fc_v5(x)
#         x = self.relu_v1(x)
#         x = self.fc_v6(x)
#         return x