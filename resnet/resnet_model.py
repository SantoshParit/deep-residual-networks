import torch
import torch.nn as nn
from .resnet_parts import BasicBlock

class ResNet20(nn.Module):
    def __init__(self, block, num_classes=10):
        super(ResNet20, self).__init__()
        self.in_channels = 16
        self.conv = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        # self.block1 = block(16, 16, stride=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        self.relu2 = nn.ReLU(inplace=True)
        
        self.conv3 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(16)
        self.shortcut1 = nn.Sequential()
        self.relu3 = nn.ReLU(inplace=True)
        
        self.block2 = block(16, 16, stride=1)
        self.block3 = block(16, 32, stride=2)
        self.block4 = block(32, 32, stride=1)
        self.block5 = block(32, 64, stride=2)
        self.block6 = block(64, 64, stride=1)

        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        
        # x = self.block1(x)
        out = self.relu2(self.bn2(self.conv2(x)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut1(x)
        out = self.relu3(out)
        
        x = self.block2(out)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
def resnet20():
    return ResNet20(BasicBlock)




