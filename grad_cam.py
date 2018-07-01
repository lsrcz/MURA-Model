#!/usr/bin/env python
# coding: utf-8
#
# Author:   Kazuto Nakashima
# URL:      http://kazuto1011.github.io
# Created:  2017-05-26
import cv2
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.datasets.folder import pil_loader
from torchvision.models import DenseNet
from torchvision.transforms import transforms

from common import device
from model import MURA_Net


class _PropagationBase(object):
    def __init__(self, model):
        super(_PropagationBase, self).__init__()
        self.device = next(model.parameters()).device
        self.model = model
        self.image = None

    def _encode_one_hot(self, idx):
        one_hot = torch.FloatTensor(1, 1).zero_()
        one_hot[0][0] = 1.0
        return one_hot.to(self.device)

    def forward(self, image):
        self.image = image.requires_grad_()
        self.model.zero_grad()
        self.preds = self.model(self.image)
        self.idx = (self.preds > 0.5).type(torch.LongTensor).numpy()[0]
        self.prob = self.preds
        return self.prob, self.idx

    def backward(self, idx):
        one_hot = self._encode_one_hot(idx)
        self.preds.backward(gradient=one_hot, retain_graph=True)


class BackPropagation(_PropagationBase):
    def generate(self):
        output = self.image.grad.detach().cpu().numpy()
        return output.transpose(0, 2, 3, 1)[0]


class GuidedBackPropagation(BackPropagation):
    def __init__(self, model):
        super(GuidedBackPropagation, self).__init__(model)

        def func_b(module, grad_in, grad_out):
            # Cut off negative gradients
            if isinstance(module, nn.ReLU):
                return (torch.clamp(grad_in[0], min=0.0), )

        for module in self.model.named_modules():
            module[1].register_backward_hook(func_b)


class Deconvolution(BackPropagation):
    def __init__(self, model):
        super(Deconvolution, self).__init__(model)

        def func_b(module, grad_in, grad_out):
            # Cut off negative gradients
            if isinstance(module, nn.ReLU):
                return (torch.clamp(grad_out[0], min=0.0), )

        for module in self.model.named_modules():
            module[1].register_backward_hook(func_b)


class GradCAM(_PropagationBase):
    def __init__(self, model):
        super(GradCAM, self).__init__(model)
        self.all_fmaps = OrderedDict()
        self.all_grads = OrderedDict()

        def func_f(module, input, output):
            self.all_fmaps[id(module)] = output.detach()

        def func_b(module, grad_in, grad_out):
            self.all_grads[id(module)] = grad_out[0].detach()

        for module in self.model.named_modules():
            print(module)
            module[1].register_forward_hook(func_f)
            module[1].register_backward_hook(func_b)

    def _find(self, outputs, target_layer):
        for key, value in outputs.items():
            for module in self.model.named_modules():
                if id(module[1]) == key:
                    if module[0] == target_layer:
                        return value
        raise ValueError('Invalid layer name: {}'.format(target_layer))

    def _normalize(self, grads):
        l2_norm = torch.sqrt(torch.mean(torch.pow(grads, 2))) + 1e-5
        return grads / l2_norm

    def _compute_grad_weights(self, grads):
        grads = self._normalize(grads)
        return F.adaptive_avg_pool2d(grads, 1)

    def generate(self, target_layer):
        fmaps = self._find(self.all_fmaps, target_layer)
        grads = self._find(self.all_grads, target_layer)
        weights = self._compute_grad_weights(grads)

        gcam = (fmaps[0] * weights[0]).sum(dim=0)
        gcam = torch.clamp(gcam, min=0.)

        gcam -= gcam.min()
        gcam /= gcam.max()

        return gcam.detach().cpu().numpy()

def grad_cam(model, image, target_layer='features.denseblock4', upsample_size=(240,240)):
    model.eval()
    gcam = GradCAM(model=model)
    probs, idx = gcam.forward(image)

    gcam.backward(idx=idx[0])
    output = gcam.generate(target_layer=target_layer)
    output = (output * 255).astype(np.uint8)
    output = cv2.resize(output, upsample_size)
    return output

def main():
    model = MURA_Net()
    model = model.to(device)
    model.load_state_dict(torch.load('./models/model.pth'))

    preprocess = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.CenterCrop(224),
        # transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    paths = [
        './MURA-v1.0/valid/XR_WRIST/patient11323/study1_positive/image1.png',
        './MURA-v1.0/valid/XR_FOREARM/patient11470/study1_positive/image2.png',
        './MURA-v1.0/valid/XR_FOREARM/patient11470/study1_positive/image3.png'
    ]

    img_pils = map(pil_loader, paths)
    img_tensors = list(map(preprocess, img_pils))

    img_variable = torch.stack(img_tensors).to(device)
    print(img_tensors[0])

    x = grad_cam(model, img_variable[2].unsqueeze(0))
    print(x)

    #o, cam = predictWithCAM(model, img_variable)
    #for i in range(len(paths)):
    #    showCAMImage(paths[i], cam[i])
    #pass

if __name__ == '__main__':
    main()
