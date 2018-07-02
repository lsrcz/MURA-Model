import torch

import cv2

import numpy as np
from PIL import Image

from transform import normalize, denormalize


def getMaxConnectedComponents(cam, width, height, threshold=0.7):
    cam = cv2.resize(cam, (width, height))
    _, bicam = cv2.threshold(cam, threshold * 255, 255, cv2.THRESH_BINARY)
    _, _, stats, _ = cv2.connectedComponentsWithStats(bicam)
    maxPos = np.argmax(stats[:, 4])
    stats[maxPos,4] = 0
    maxPos = np.argmax(stats[:, 4])
    left = stats[maxPos, 0]
    top = stats[maxPos, 1]
    width = stats[maxPos, 2]
    height = stats[maxPos, 3]
    area = stats[maxPos, 4]
    return left, top, width, height, area

def add_heatmap(cam, img, need_transpose_color=True):
    img = np.array(img)
    height, width, _ = img.shape
    cam = cv2.resize(cam, (width, height))
    heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    result = (heatmap * 0.2 + img * 0.75).astype(np.uint8)
    if need_transpose_color:
        result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    return result

def add_heatmap_ts(cam, tsimg, need_transpose_color=True):
    return add_heatmap(cam, denormalize(tsimg.cpu().detach()), need_transpose_color)

def add_boundedbox(cam, img, threshold=0.7, need_transpose_color=False):
    img = np.array(img)
    height, width, _ = img.shape
    left, top, width, height, area = getMaxConnectedComponents(cam, width, height)

    img = cv2.line(img, (left, top), (left + width, top), (0,0,255),2)
    img = cv2.line(img, (left, top), (left, top + height), (0,0,255),2)
    img = cv2.line(img, (left, top + height), (left + width, top + height), (0,0,255),2)
    img = cv2.line(img, (left + width, top), (left + width, top + height), (0,0,255),2)
    if need_transpose_color:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def add_boundedbox_ts(cam, tsimg, threshold=0.7, need_transpose_color=True):
    return add_boundedbox(cam, denormalize(tsimg.cpu().detach()), threshold, need_transpose_color)

# don't know if it's correct
def crop_heat(cams, tsimgs, threshold=0.7):
    bs = tsimgs.shape[0]
    arr = []
    for i in range(bs):
        img = np.array(denormalize(tsimgs[i].detach().cpu()))
        height, width, _ = img.shape
        left, top, width, height, area = getMaxConnectedComponents(cams[i], height, width, threshold)
        img = img[top:top+height,left:left+width+1]
        img = cv2.resize(img, (224,224))
        img = Image.fromarray(img)
        arr.append(normalize(img))
    return torch.stack(arr)

def tsimg2img(tsimg):
    return denormalize(tsimg.cpu().detach())

'''
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

    path = './MURA-v1.0/valid/XR_WRIST/patient11285/study1_positive/image1.png'

    img_pil = pil_loader(path)
    img_tensor = preprocess(img_pil)

    img_variable = torch.stack([img_tensor]).to(device)

    x = grad_cam(model, img_variable)

    t = crop_heat([x], img_variable)
    plt.imshow(_denormalize(img_variable[0].cpu().detach()))
    plt.show()
    plt.imshow(_denormalize(t[0]))
    plt.show()

    img = _denormalize(img_variable[0].cpu().detach())
    img = add_boundedbox(x, img)
    print(img)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()

    plt.imshow(add_heatmap(x, img))
    plt.show()


if __name__ == '__main__':
    main()
'''
