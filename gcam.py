import cv2

import torch.nn.functional as F
import torch
from torchvision.datasets.folder import pil_loader
from torchvision.transforms import transforms

from common import device
import numpy as np

# hook the feature extractor
from model import MURA_Net

features_blobs = []
def hook_feature(module, input, output):
    features_blobs.append(output.detach())

grad_blobs = []
def hook_grad(module, grad_in, grad_out):
    grad_blobs.append(grad_out[0].detach())

def _l2_normalize(grads):
    print(grads.shape)
    l2_norm = torch.sqrt(torch.mean(torch.mean(
        torch.mean(torch.pow(grads, 2), dim=3), dim=2), dim=1)) + 1e-5
    print(l2_norm)
    return grads / l2_norm.reshape(grads.shape[0], 1, 1, 1)

def _normalize(grads):
    print(grads.shape)
    l2_norm = torch.sqrt(torch.mean(torch.pow(grads, 2))) + 1e-5
    print(l2_norm)
    return grads / l2_norm

def _one_hot(bs):
    one_hot = torch.ones(bs ,1)
    return one_hot.to(device)

def _min_inside(gcam):
    m, _ = torch.min(gcam, dim=2)
    m, _ = torch.min(m, dim=1)
    return m

def _max_inside(gcam):
    m, _ = torch.max(gcam, dim=2)
    m, _ = torch.max(m, dim=1)
    return m

def gcam(model, img, name="features.denseblock4", upsample_size=(224,224)):
    model.eval()
    namelst = name.split('.')
    module = model
    for name in namelst:
        module = module._modules.get(name)
    module.register_forward_hook(hook_feature)
    module.register_backward_hook(hook_grad)
    img.requires_grad_()
    model.zero_grad()

    preds = model(img)
    fmaps = features_blobs[-1]
    probs = F.sigmoid(preds)

    # prob = preds
    one_hot = _one_hot(probs.shape[0])
    preds.backward(gradient=one_hot, retain_graph=True)


    grads = grad_blobs[-1]
    grads = _l2_normalize(grads)
    weights = F.adaptive_avg_pool2d(grads ,1)
    gcam = (fmaps * weights).sum(dim=1)
    gcam = torch.clamp(gcam, min=0.)

    print(gcam.shape)
    print(_min_inside(gcam).shape)
    gcam -= _min_inside(gcam).reshape(-1, 1, 1)
    gcam /= _max_inside(gcam).reshape(-1, 1, 1)


    output = (gcam.detach().cpu().numpy() * 255).astype(np.uint8)
    output = output.transpose(1 ,2 ,0)
    output = cv2.resize(output, (224, 224)).transpose(2 ,0 ,1)
    return output

_preprocess = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.CenterCrop(224),
    # transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def main():
    model = MURA_Net()
    model = model.to(device)
    model.load_state_dict(torch.load('./models/model_50.pth'))

    paths = ['./MURA-v1.0/valid/XR_SHOULDER/patient11791/study1_positive/image1.png',
             './MURA-v1.0/valid/XR_SHOULDER/patient11791/study1_positive/image2.png',
             './MURA-v1.0/valid/XR_SHOULDER/patient11791/study1_positive/image3.png',
             './MURA-v1.0/valid/XR_SHOULDER/patient11791/study1_positive/image4.png'] * 15
    img_pil = list(map(pil_loader, paths))

    img_tensor = list(map(_preprocess, img_pil))
    img_variable = torch.stack(img_tensor).to(device)

    colormap = gcam(model, img_variable)
    print(colormap)

if __name__ == '__main__':
    main()
