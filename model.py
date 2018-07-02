import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import torchvision
from torchvision import transforms, models

from common import device
from gcam import gcam
from localize import crop_heat


class MURA_Net(nn.Module):
    def __init__(self, networkName='densenet161'):
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
    def __init__(self, networkName='densenet161'):
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

class MURA_Net_AG(nn.Module):
    def __init__(self, networkName='densenet161'):
        assert networkName in ['densenet169','densenet161']
        self.networkName = networkName
        super(MURA_Net_AG, self).__init__()
        if networkName == 'densenet169':
            self.global_net = MURA_Net(networkName)#torchvision.models.densenet169(pretrained=True)
            self.local_net = MURA_Net(networkName)#torchvision.models.densenet169(pretrained=True)
            self.classifier = nn.Linear(1664 * 2, 1)
        if networkName == 'densenet161':
            self.global_net = MURA_Net(networkName)  # torchvision.models.densenet169(pretrained=True)
            self.local_net = MURA_Net(networkName)  # torchvision.models.densenet169(pretrained=True)
            self.classifier = nn.Linear(2208 * 2, 1)

    def load_global_dict(self, global_dict):
        self.global_net.load_state_dict(global_dict)

    def load_local_dict(self, local_dict):
        self.local_net.load_state_dict(local_dict)

    def forward(self, input):
        if self.networkName == 'densenet169':
            global_features = self.global_net.features(input)
            global_features = F.relu(global_features, inplace=True)
            global_features = F.avg_pool2d(global_features, kernel_size=7, stride=1)\
                .view(self.global_net.features.size[0], -1)

            cams = gcam(self, input)
            local_input = crop_heat(cams, input).to(device)

            local_features = self.local_net.features(local_input)
            local_features = F.relu(local_features, inplace=True)
            local_features = F.avg_pool2d(local_features, kernel_size=7, stride=1) \
                .view(self.local_net.features.size[0], -1)

            out = torch.cat([global_features, local_features], dim=1)
            out = self.classifier(out)
            out = F.softmax(out)
            return out

        if self.networkName == 'densenet161':
            print(1)
            global_features = self.global_net.features(input)
            print(2)
            global_features = F.relu(global_features, inplace=True)
            print(3)
            global_features = F.avg_pool2d(global_features, kernel_size=7, stride=1) \
                .view(self.global_net.features.size[0], -1)
            print(4)

            cams = gcam(self, input)
            local_input = crop_heat(cams, input).to(device)

            local_features = self.local_net.features(local_input)
            local_features = F.relu(local_features, inplace=True)
            local_features = F.avg_pool2d(local_features, kernel_size=7, stride=1) \
                .view(self.local_net.features.size[0], -1)

            out = torch.cat([global_features, local_features], dim=1)
            out = self.classifier(out)
            out = F.softmax(out)
            return out




def main():
    x = MURA_Net()
    i = 0

if __name__ == '__main__':
    main()