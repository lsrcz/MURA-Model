import torch
from common import *
import numpy as np
import matplotlib.pyplot as plt


class AUCMeter:

    def __init__(self, label, color, linestyle):
        self.label = label
        self.color = color
        self.linestyle = linestyle
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
        plt.plot(fpr, tpr, label=self.label, color=self.color, linestyle=self.linestyle)
        plt.title("AUC curve: " + self.label)
        plt.xlabel('fpr')
        plt.ylabel('tpr')
        plt.xlim(0,1)
        plt.ylim(0,1)
        plt.text(0.5, 0.5, 'area: ' + str(area))
        plt.show()

class AUCMeterMulti:

    def __init__(self):
        self.meters = {}

    def add_meter(self, label, color, linestyle):
        assert not label in self.meters
        self.meters[label] = AUCMeter(label, color, linestyle)

    def __len__(self):
        return len(self.meters)

    def __getitem__(self, item):
        return self.meters[item]

    def plot(self):
        allstr = "Areas:\n"
        fig = plt.figure(figsize=(12,9), dpi=180)
        ax = fig.gca()
        ax.set_title("AUC curves")
        ax.set_xlabel('fpr')
        ax.set_ylabel('tpr')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')

        for name in self.meters.keys():
            meter = self.meters[name]
            area, tpr, fpr = meter.value()
            ax.plot(fpr, tpr, linewidth=1, label=meter.label, color=meter.color, linestyle=meter.linestyle)
            allstr += (name + ": " + ("%.4f" % area) + "\n")
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.text(0.5, 0.5, allstr)
        plt.show()

class ConfusionMatrixMeter():

    def __init__(self):
        self.conf = np.ndarray((2, 2), dtype=np.int32)
        self.reset()

    def reset(self):
        self.conf.fill(0)

    def add(self, pred, target):
        if torch.is_tensor(pred):
            pred = pred.cpu().numpy()
        if torch.is_tensor(target):
            target = target.cpu().numpy()

        assert pred.shape[0] == target.shape[0], \
            'number of targets and predicted outputs do not match'

        if np.ndim(pred) != 1:
            assert pred.shape[1] == 2, \
                'number of predictions does not match size of confusion matrix'
            pred = np.argmax(pred, 1)
        else:
            assert (pred.max() == 1) or (pred.min() == 0), \
                'predicted values are not 0 or 1'

        onehot_target = np.ndim(target) != 1
        if onehot_target:
            assert target.shape[1] == 2, \
                'Onehot target does not match size of confusion matrix'
            assert (target >= 0).all() and (target <= 1).all(), \
                'in one-hot encoding, target values should be 0 or 1'
            assert (target.sum(1) == 1).all(), \
                'multi-label setting is not supported'
            target = np.argmax(target, 1)
        else:
            assert (pred.max() <= 1) and (pred.min() >= 0), \
                'predicted values are not 0 or 1'

        # hack for bincounting 2 arrays together
        x = pred + 2 * target
        bincount_2d = np.bincount(x.astype(np.int32),
                                  minlength=2 ** 2)
        assert bincount_2d.size == 2 ** 2
        conf = bincount_2d.reshape((2, 2))

        self.conf += conf

    def value(self):
        return self.conf

    def TP(self):
        return self.conf[1,1]

    def FP(self):
        return self.conf[0,1]

    def TN(self):
        return self.conf[0,0]

    def FN(self):
        return self.conf[1,0]

    def precision(self):
        return float(self.TP()) / (self.TP() + self.FP())

    def recall(self):
        return float(self.TP()) / (self.TP() / self.FN())

    def TPR(self):
        return self.precision()

    def FPR(self):
        return self.FP() / (self.FP() + self.TN())

    def F1(self):
        return float(2 * self.TP()) / (2.0 * self.TP() + self.FP() + self.FN())

    def count(self):
        return self.TP() + self.TN() + self.FN() + self.FP()

    def accuracy(self):
        return float(self.TP() + self.TN()) / self.count()

    def kappa(self):
        observed = self.accuracy()
        exp1 = float((self.TN() + self.FN()) * (self.TN() + self.FP())) / self.count()
        exp2 = float((self.TP() + self.FN()) * (self.TP() + self.FP())) / self.count()
        expected = (exp1 + exp2) / self.count()
        return (observed - expected) / (1 - expected)

class ConfusionMatrixMeterMulti():

    def _call_sub(self, name, study_name=None):
        if study_name:
            return getattr(self.meters[study_name], name)()
        return getattr(self.meters['#tot#'], name)()

    def _init_sub(self):
        for attr in ['value', 'TP', 'FP', 'TN', 'FN',
                     'precision', 'recall', 'TPR', 'FPR',
                     'F1', 'count', 'accuracy', 'kappa']:
            def attrbdy(name):
                def closure(study_name=None):
                    return self._call_sub(name, study_name)
                return closure
            setattr(self, attr, attrbdy(attr))

    def __init__(self):
        self.reset()
        self._init_sub()

    def reset(self):
        self.meters = {'#tot#': ConfusionMatrixMeter()}
        self.names = set()

    def add(self, pred, target, study_names):
        study_names = np.array(study_names)
        pred = pred.cpu().numpy()
        target = target.cpu().numpy()

        assert pred.shape[0] == target.shape[0], \
            'number of targets and predicted outputs do not match'

        if np.ndim(pred) != 1:
            assert pred.shape[1] == 2, \
                'number of predictions does not match size of confusion matrix'
            pred = np.argmax(pred, 1)
        else:

            assert (pred.max() <= 1) and (pred.min() >= 0), \
                str(pred) + '\npredicted values are not 0 or 1'

        onehot_target = np.ndim(target) != 1
        if onehot_target:
            assert target.shape[1] == 2, \
                'Onehot target does not match size of confusion matrix'
            assert (target >= 0).all() and (target <= 1).all(), \
                'in one-hot encoding, target values should be 0 or 1'
            assert (target.sum(1) == 1).all(), \
                'multi-label setting is not supported'
            target = np.argmax(target, 1)
        else:
            assert (pred.max() < 2) and (pred.min() >= 0), \
                'predicted values are not 0 or 1'

        for name in np.unique(study_names):
            if name not in self.meters:
                self.meters[name] = ConfusionMatrixMeter()
                self.names.add(name)
            self.meters[name].add(pred[study_names == name], target[study_names == name])

        self.meters['#tot#'].add(pred, target)

def main():
    c = ConfusionMatrixMeterMulti()
    c.add(torch.Tensor([0] * 10 + [1] * 5 + [0] * 7 + [1] * 8), torch.Tensor([0] * 15 + [1] * 15), ['a'] * 30)
    print(c.value())

if __name__ == '__main__':
    main()




