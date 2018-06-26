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

net._modules.get(finalconv_name).register_forward_hook(hook_feature)

# get the softmax weight
params = list(net.parameters())
weight_softmax = np.squeeze(params[-2].data.cpu().numpy())
#print(np.shape(weight_softmax))
def returnCAM(feature_conv, weight_softmax):
    # generate the class activation maps upsample to 256x256
    size_upsample = (320, 320)
    bz, nc, h, w = feature_conv.shape
    print(bz,nc,h,w)
    print(np.shape(weight_softmax))
    output_cam = []
    cam = weight_softmax.dot(feature_conv.reshape((nc, h*w)))
    cam = cam.reshape(h, w)
    cam = cam - np.min(cam)
    cam_img = cam / np.max(cam)
    cam_img = np.uint8(255 * cam_img)
    print(cam_img)
    output_cam.append(cv2.resize(cam_img, size_upsample))
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
preds = (outputs > 0.5).type(torch.LongTensor).numpy()[0]

# generate class activation mapping for the top1 prediction
CAMs = returnCAM(features_blobs[0], weight_softmax)


# render the CAM and output
print('output CAM.jpg for the top1 prediction: %d'%preds)
img = cv2.imread('./MURA-v1.0/valid/XR_FOREARM/patient11470/study1_positive/image1.png')
height, width, _ = img.shape
heatmap = cv2.applyColorMap(cv2.resize(CAMs[0],(width, height)), cv2.COLORMAP_JET)
result = heatmap * 0.3 + img * 0.5
cv2.imwrite('CAM.jpg', result)
img = Image.open('CAM.jpg')
img.show()