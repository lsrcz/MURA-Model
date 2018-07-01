import torch
import numpy as np
import cv2
from model import MURA_Net
from common import *
from torchvision import models, transforms
from torchvision.datasets.folder import pil_loader
from PIL import Image
import matplotlib.pyplot as plt

features_blobs = []

def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())

def predictWithCAM(model, data, upsample_size=(320, 320)):
    global features_blobs
    hook = model._modules.get('features').register_forward_hook(hook_feature)
    params = list(model.parameters())
    weight_softmax = np.squeeze(params[-2].data.cpu().numpy())

    outputs = model(data)

    bz, nc, h, w = features_blobs[-1].shape
    print(bz, nc, h, w)
    print(np.shape(weight_softmax))
    output_cam = []
    cam = weight_softmax.dot(features_blobs[-1].reshape((bz, nc, h * w)))
    cam = cam.reshape(bz, h, w)
    cam = cam - np.min(cam, axis=(1,2)).reshape(bz,1,1)
    cam_img = cam / np.max(cam, axis=(1,2)).reshape(bz,1,1)
    cam_img = np.uint8(255 * cam_img)
    print(cam_img)
    cam_img = cam_img.transpose(1,2,0)
    hook.remove()
    features_blobs = []
    return outputs, cv2.resize(cam_img, upsample_size).transpose(2,0,1)

def showCAMImage(path, colormap):
    img = cv2.imread(path)
    height, width, _ = img.shape
    heatmap = cv2.applyColorMap(cv2.resize(colormap, (width, height)), cv2.COLORMAP_JET)
    result = (heatmap * 0.3 + img * 0.5).astype(np.uint8)

    resultimg = Image.fromarray(cv2.cvtColor(result,cv2.COLOR_BGR2RGB))
    plt.figure(figsize=(12,9),dpi=180)
    plt.imshow(resultimg)
    plt.show()

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
        './MURA-v1.0/valid/XR_FOREARM/patient11470/study1_positive/image1.png',
        './MURA-v1.0/valid/XR_FOREARM/patient11470/study1_positive/image2.png',
        './MURA-v1.0/valid/XR_FOREARM/patient11470/study1_positive/image3.png'
    ]

    img_pils = map(pil_loader, paths)
    img_tensors = list(map(preprocess, img_pils))

    img_variable = torch.stack(img_tensors).to(device)
    o, cam = predictWithCAM(model, img_variable)
    for i in range(len(paths)):
        showCAMImage(paths[i], cam[i])



if __name__ == '__main__':
    main()