import torch.nn as nn
import torch.nn.init as init

class GUIDE_FCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(77, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.relu1 = nn.ReLU()

        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu2 = nn.ReLU()

        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.relu3 = nn.ReLU()

        self.cls1 = nn.Linear(128, 64)
        self.cls_bn = nn.BatchNorm1d(64)
        self.relu4 = nn.ReLU()
        self.cls2 = nn.Linear(64, 3)

        self._initialize_weights()

    def forward(self, x):
        x = x.view(x.size(0), -1)

        x = self.relu1(self.bn1(self.fc1(x)))
        x = self.relu2(self.bn2(self.fc2(x)))
        x = self.relu3(self.bn3(self.fc3(x)))

        x = self.relu4(self.cls_bn(self.cls1(x)))
        return self.cls2(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                init.constant_(m.bias, 0.01)
