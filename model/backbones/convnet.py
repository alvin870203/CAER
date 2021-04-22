import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Block(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, padding = 1):
        super(Block, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding, bias=False)
        self.bn   = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x

class ConvNet(nn.Module):

    def __init__(self):
        super(ConvNet, self).__init__()
        
        self.pool = nn.MaxPool2d(2, 2)
        
        self.block_1 = Block(3, 32, 1, 1)
        self.block_2 = Block(32, 64, 1, 1)
        self.block_3 = Block(64, 128, 1, 1)
        self.block_4 = Block(128, 256, 1, 1)
        self.block_5 = Block(256, 256, 1, 1)
        
        self.random_init()

    def forward(self, x):
        x = self.block_1(x)
        x = self.pool(x)

        x = self.block_2(x)
        x = self.pool(x)


        x = self.block_3(x)
        x = self.pool(x)


        x = self.block_4(x)
        x = self.pool(x)


        x = self.block_5(x)
        return x

    def random_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()