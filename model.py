import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import torchvision
from torchvision import transforms, models

from common import device
from grad_cam import grad_cam_batch
from localize import crop_heat


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
            self.global_net = MURA_Net(networkName)#torchvision.models.densenet169(pretrained=True)
            self.local_net = MURA_Net(networkName)#torchvision.models.densenet169(pretrained=True)
            self.classifier = nn.Linear(1664 * 2, 1)

    def load_global_dict(self, global_dict):
        self.global_net.load_state_dict(global_dict)

    def load_local_dict(self, local_dict):
        self.local_net.load_state_dict(local_dict)

    def forward(self, input):
        if self.networkName == 'densenet169':
            global_features = self.global_net.features(input)
            global_features = F.relu(global_features, inplace=True)
            global_features = F.avg_pool2d(global_features, kernel_size=7, stride=1)\
                .view(self.global_net.features.size(0), -1)

            cams = grad_cam_batch(self, input)
            local_input = crop_heat(cams, input).to(device)

            local_features = self.local_net.features(local_input)
            local_features = F.relu(local_features, inplace=True)
            local_features = F.avg_pool2d(local_features, kernel_size=7, stride=1) \
                .view(self.local_net.features.size(0), -1)

            out = torch.cat([global_features, local_features], dim=1)
            out = self.classifier(out)
            out = F.softmax(out)
            return out




def main():
    x = MURA_Net()
    i = 0

if __name__ == '__main__':
    main()