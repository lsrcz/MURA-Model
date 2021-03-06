import cv2

import torch.nn.functional as F
import torch
from torchvision.transforms import transforms
from common import device
import numpy as np

# hook the feature extractor

def _l2_normalize(grads):
    l2_norm = torch.sqrt(torch.mean(torch.mean(
        torch.mean(torch.pow(grads, 2), dim=3), dim=2), dim=1)) + 1e-5
    return grads / l2_norm.reshape(grads.shape[0], 1, 1, 1)

def _normalize(grads):
    l2_norm = torch.sqrt(torch.mean(torch.pow(grads, 2))) + 1e-5
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
    features_blobs = [None]

    def hook_feature(module, input, output):
        features_blobs[0] = output.detach()

    grad_blobs = [None]

    def hook_grad(module, grad_in, grad_out):
        grad_blobs[0] = grad_out[0].detach()

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
    fmaps = features_blobs[0]
    probs = F.sigmoid(preds)

    # prob = preds
    one_hot = _one_hot(probs.shape[0])
    preds.backward(gradient=one_hot, retain_graph=False)

    grads = grad_blobs[0]
    grads = _l2_normalize(grads)
    weights = F.adaptive_avg_pool2d(grads ,1)
    gcam = (fmaps * weights).sum(dim=1)
    gcam = torch.clamp(gcam, min=0.)

    gcam -= _min_inside(gcam).reshape(-1, 1, 1)
    gcam /= _max_inside(gcam).reshape(-1, 1, 1)

    output = (gcam.detach().cpu().numpy() * 255).astype(np.uint8)
    output = output.transpose(1 ,2 ,0)
    output = cv2.resize(output, (224, 224))
    if output.ndim == 2:
        output = output.reshape(1, 224, 224)
    else:
        output = output.transpose(2 ,0 ,1)
    return preds, output

_preprocess = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.CenterCrop(224),
    # transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

