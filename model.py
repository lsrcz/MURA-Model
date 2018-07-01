import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import torchvision
from torchvision import transforms, models

class MURA_Net(nn.Module):
    def __init__(self, networkName='densenet169'):
        assert networkName in ['densenet169']
        super(MURA_Net, self).__init__()
        if networkName == 'densenet169':
            self.features = torchvision.models.densenet169(pretrained=True).features
            self.classifier = nn.Linear(1664, 1)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.avg_pool2d(out, kernel_size=7, stride=1).view(features.size(0), -1)
        out = self.classifier(out)
        out = F.sigmoid(out)
        return out


class MURA_Net_Binary(nn.Module):
    def __init__(self, networkName='densenet169'):
        assert networkName in ['densenet169']
        super(MURA_Net_Binary, self).__init__()
        if networkName == 'densenet169':
            self.features = torchvision.models.densenet169(pretrained=True).features
            self.classifier = nn.Linear(1664, 2)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.avg_pool2d(out, kernel_size=7, stride=1).view(features.size(0), -1)
        out = self.classifier(out)
        out = F.softmax(out,dim=1)
        return out

class MURA_Net_AG(nn.Module):
    def __init__(self, networkName='densenet169'):
        assert networkName in ['densenet169']
        self.networkName = networkName
        super(MURA_Net_AG, self).__init__()
        if networkName == 'densenet169':
            self.global_net = torchvision.models.densenet169(pretrained=True)
            self.local_net = torchvision.models.densenet169(pretrained=True)
            self.classifier = nn.Linear(1664 * 2, 1)

    def forward(self, x):
        global_features = self.global_net.features(x)
        # TODO: finish the forward



def main():
    x = MURA_Net()
    i = 0

if __name__ == '__main__':
    main()