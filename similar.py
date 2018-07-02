from common import *
from tqdm import tqdm
import operator
from dataset import get_dataloaders
from model import MURA_Net, MURA_Net_Binary
import numpy as np
from torchvision import models, transforms
from torchvision.datasets.folder import pil_loader
import cv2
from PIL import Image
import matplotlib.pyplot as plt

similarity = {}

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
    features = np.mean(features_blobs[-1], axis = (2,3))
    for i in range(features.shape[0]):
        features[i] = weight_softmax*(features[i])
    hook.remove()
    features_blobs = []
    return outputs, features

def find_img_tensor(dataloaders, path):
    for i, data in enumerate(dataloaders['valid']):
        global img_tensors
        images = data['image']['norm']
        metadatas = data['metadata']
        paths = metadatas['path']
        for j in range(images.shape[0]):
            if paths[j] == path:
                img_tensors = [images[j],images[j]]
                return img_tensors

def findTOP5(model, dataloaders, path):
    model.eval()

    preprocess = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.CenterCrop(224),
        # transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    paths = [path,path]
    img_pils = map(pil_loader, paths)
    img_tensors = list(map(preprocess, img_pils))

    img_variable = torch.stack(img_tensors).to(device)
    img = cv2.imread(path)

    resultimg = Image.fromarray(img)
    plt.figure(figsize=(12, 9), dpi=180)
    plt.imshow(resultimg)
    plt.show()
    with torch.no_grad():
        out, features_ori = predictWithCAM(model, img_variable)
    features_ori = features_ori[0]
    for i, data in enumerate(tqdm(dataloaders['valid'])):
        images = data['image']['norm']
        labels = data['label']
        metadatas = data['metadata']
        paths = metadatas['path']

        images = images.to(device)
        with torch.no_grad():
            o, features = predictWithCAM(model, images)
            for j in range(images.shape[0]):
                similarity[paths[j]] = np.linalg.norm(features[j] - features_ori, ord=1)

    sim = sorted(similarity.items(), key=operator.itemgetter(1), reverse=False)
    for i in range(0,5):
        path = sim[i][0]
        print(path)
        img = cv2.imread(path)

        resultimg = Image.fromarray(img)
        plt.figure(figsize=(12, 9), dpi=180)
        plt.imshow(resultimg)
        plt.show()




def main():
    dataloaders, dataset_sizes = get_dataloaders(
        study_name=None,
        data_dir='MURA-v1.0',
        batch_size=50,
        batch_eval_ten=15,
        shuffle=False
    )

    model = MURA_Net()
    model = model.to(device)
    model.load_state_dict(torch.load('models/model_densenet161_fixed.pth'))

    img_path = 'MURA-v1.0/valid/XR_FOREARM/patient11470/study1_positive/image3.png'

    #img_path = 'MURA-v1.0/train/XR_FOREARM/patient09351/study1_positive/image2.png'

    #img_path = 'MURA-v1.0/train/XR_FOREARM/patient09356/study1_positive/image1.png'

    #img_path = 'MURA-v1.0/train/XR_SHOULDER/patient01293/study1_positive/image1.png'
    findTOP5(model, dataloaders, img_path)

if __name__ == '__main__':
    main()

