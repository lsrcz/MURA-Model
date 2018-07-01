import io
import requests
from model import MURA_Net
from common import *
from PIL import Image
from torchvision import models, transforms
from torchvision.datasets.folder import pil_loader
from torch.nn import functional as F
import torch
import numpy as np
import cv2
# networks such as googlenet, resnet, densenet already use global average pooling at the end, so CAM could be used directly.
net = MURA_Net()
net = net.to(device)
net.load_state_dict(torch.load('./models/model.pth'))
finalconv_name = 'features'

net.eval()

# hook the feature extractor
features_blobs = []
def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())

grad_blobs = []
def hook_grad(module, grad_in, grad_out):
    grad_blobs.append(grad_out[0].data.cpu().numpy())

net._modules.get(finalconv_name).register_forward_hook(hook_feature)
net._modules.get(finalconv_name).register_backward_hook(hook_grad)

# get the softmax weight
params = list(net.parameters())
weight_softmax = np.squeeze(params[-2].data.cpu().numpy())
#print(np.shape(weight_softmax))
def returnCAM(feature_conv, grad_conv):
    # generate the class activation maps upsample to 256x256
    feature_conv = np.squeeze(feature_conv, axis=0)
    grad_conv = np.squeeze(grad_conv, axis=0)
    size_upsample = (320, 320)
    #print(feature_conv.shape)
    weights = np.mean(grad_conv, axis=(1,2),keepdims = True)
    #print(weights)
    output_cam = []
    cam = np.sum(weights * feature_conv, axis = 0)
    cam = cam - np.min(cam)
    cam_img = cam / (np.max(cam)-np.min(cam))
    cam_img = 1.0 - cam_img
    cam_img = np.uint8(255 * cam_img)
    print(cam_img.shape)
    output_cam.append(cv2.resize(cam_img, size_upsample))
    print(len(output_cam))
    return output_cam


preprocess = transforms.Compose([
            transforms.Resize((320, 320)),
            transforms.CenterCrop(224),
            #transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

img_pil = pil_loader('./MURA-v1.0/valid/XR_FOREARM/patient11470/study1_positive/image1.png')

img_tensor = preprocess(img_pil)
img_variable = img_tensor.unsqueeze(0).to(device)
outputs = net(img_variable)
preds = (outputs.data > 0.5)

net.zero_grad()

loss = F.binary_cross_entropy(outputs, preds.float())
loss.backward()

# generate class activation mapping for the top1 prediction
print(features_blobs[0], grad_blobs[0])
CAMs = returnCAM(features_blobs[0], grad_blobs[0])


# render the CAM and output
print('output CAM.jpg for the top1 prediction: %d'%preds)
img = cv2.imread('./MURA-v1.0/valid/XR_FOREARM/patient11470/study1_positive/image1.png')
height, width, _ = img.shape
heatmap = cv2.applyColorMap(cv2.resize(CAMs[0],(width, height)), cv2.COLORMAP_JET)
result = heatmap * 0.3 + img * 0.5
cv2.imwrite('./out/gradCAM.jpg', result)
img = Image.open('./out/gradCAM.jpg')
img.show()