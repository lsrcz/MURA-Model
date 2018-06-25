import torch
from common import *
import numpy as np
import matplotlib.pyplot as plt

class AUCMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.scores = torch.DoubleTensor(torch.DoubleStorage()).numpy()
        self.targets = torch.LongTensor(torch.LongStorage()).numpy()

    def add(self, output, target):
        if torch.is_tensor(output):
            output = output.to(cpu_device).squeeze().numpy()
        if torch.is_tensor(target):
            target = target.to(cpu_device).squeeze().numpy()
        self.scores = np.append(self.scores, output)
        self.targets = np.append(self.targets, target)

    def value(self):
        if self.scores.shape[0] == 0:
            return 0.5

        # sorting the arrays
        scores, sortind = torch.sort(torch.from_numpy(
            self.scores), dim=0, descending=True)
        scores = scores.numpy()
        sortind = sortind.numpy()

        # creating the roc curve
        tpr = np.zeros(shape=(scores.size + 1), dtype=np.float64)
        fpr = np.zeros(shape=(scores.size + 1), dtype=np.float64)

        for i in range(1, scores.size + 1):
            if self.targets[sortind[i - 1]] == 1:
                tpr[i] = tpr[i - 1] + 1
                fpr[i] = fpr[i - 1]
            else:
                tpr[i] = tpr[i - 1]
                fpr[i] = fpr[i - 1] + 1

        tpr /= (self.targets.sum() * 1.0)
        fpr /= ((self.targets - 1.0).sum() * -1.0)

        # calculating area under curve using trapezoidal rule
        n = tpr.shape[0]
        h = fpr[1:n] - fpr[0:n - 1]
        sum_h = np.zeros(fpr.shape)
        sum_h[0:n - 1] = h
        sum_h[1:n] += h
        area = (sum_h * tpr).sum() / 2.0

        return area, tpr, fpr

    def plot(self):
        area, tpr, fpr = self.value()
        plt.plot(fpr, tpr)
        plt.title("AUC curve")
        plt.xlabel('fpr')
        plt.ylabel('tpr')
        plt.text(0.5, 0.5, 'area: ' + str(area))
        plt.show()
