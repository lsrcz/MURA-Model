import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import torchvision
from torchvision import transforms, models


class MURA_Net(nn.Module):
    def __init__(self, networkName='densenet169'):
        assert networkName in ['densenet169','densenet161','densenet201']
        super(MURA_Net, self).__init__()
        if networkName == 'densenet169':
            self.features = torchvision.models.densenet169(pretrained=True).features
            self.classifier = nn.Linear(1664, 1)
        if networkName == 'densenet161':
            self.features = torchvision.models.densenet161(pretrained=True).features
            self.classifier = nn.Linear(2208, 1)
        if networkName == 'densenet201':
            self.features = torchvision.models.densenet201(pretrained=True).features
            self.classifier = nn.Linear(1920, 1)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.avg_pool2d(out, kernel_size=7, stride=1).view(features.size(0), -1)
        out = self.classifier(out)
        out = F.sigmoid(out)
        return out


class MURA_Net_Binary(nn.Module):
    def __init__(self, networkName='densenet169'):
        assert networkName in ['densenet169', 'densenet161', 'densenet201']
        super(MURA_Net_Binary, self).__init__()
        if networkName == 'densenet169':
            self.features = torchvision.models.densenet169(pretrained=True).features
            self.classifier = nn.Linear(1664, 2)
        if networkName == 'densenet161':
            self.features = torchvision.models.densenet161(pretrained=True).features
            self.classifier = nn.Linear(2208, 2)
        if networkName == 'densenet201':
            self.features = torchvision.models.densenet201(pretrained=True).features
            self.classifier = nn.Linear(1920, 2)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.avg_pool2d(out, kernel_size=7, stride=1).view(features.size(0), -1)
        out = self.classifier(out)
        out = F.softmax(out,dim=1)
        return out


def main():
    x = MURA_Net()
    i = 0

if __name__ == '__main__':
    main()