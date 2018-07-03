from common import *
from optparse import OptionParser
from tqdm import tqdm
import operator
from dataset import get_dataloaders
from model import MURA_Net, get_pretrained_model
import numpy as np
from torchvision import transforms
from torchvision.datasets.folder import pil_loader
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from gcam import gcam
from localize import add_heatmap_ts

similarity = {}

features_blobs = []


def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())

def getFeature(model, data, upsample_size=(320, 320)):
    global features_blobs
    hook = model._modules.get('features').register_forward_hook(hook_feature)
    params = list(model.parameters())
    weight_softmax = np.squeeze(params[-2].data.cpu().numpy())

    outputs = model(data)

    bz, nc, h, w = features_blobs[-1].shape
    features = np.mean(features_blobs[-1], axis=(2, 3))
    for i in range(features.shape[0]):
        features[i] = weight_softmax * (features[i])
    hook.remove()
    features_blobs = []
    return outputs, features

preprocess = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.CenterCrop(224),
    # transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def cross_entropy(a, y):
    return -np.sum(np.nan_to_num(y*np.log(a)+(1-y)*np.log(1-a)))

def findTOP5addr(model, dataloaders, path):
    model.eval()

    paths = [path, path]
    img_pils = map(pil_loader, paths)
    img_tensors = list(map(preprocess, img_pils))

    img_variable = torch.stack(img_tensors).to(device)

    with torch.no_grad():
        out, features_ori = getFeature(model, img_variable)
    features_ori = features_ori[0]
    for i, data in enumerate(tqdm(dataloaders['valid'])):
        images = data['image']['norm']
        metadatas = data['metadata']
        paths = metadatas['path']

        images = images.to(device)
        with torch.no_grad():
            o, features = getFeature(model, images)
            for j in range(images.shape[0]):
                similarity[paths[j]] = np.linalg.norm(features[j] - features_ori)

    sim = sorted(similarity.items(), key=operator.itemgetter(1), reverse=False)
    path = []
    for i in range(5):
        path.append(sim[i][0])
        print(sim[i][0])
    return path

def findTOP5pic(model, dataloaders, path):
    path = findTOP5addr(model, dataloaders, path)

    img_pils = map(pil_loader, path)
    img_tensors = list(map(preprocess, img_pils))

    img_variable = torch.stack(img_tensors).to(device)
    output, heatmap = gcam(model, img_variable, upsample_size=(320, 320))
    for i in range(0, 5):
        plt.imshow(add_heatmap_ts(heatmap[i], img_variable[i]))
        plt.show()

        img = cv2.imread(path[i])

        resultimg = Image.fromarray(img)
        plt.figure(figsize=(12, 9), dpi=180)
        plt.imshow(resultimg)
        plt.show()

def main():

    usage = "usage: %prog [option] img_path dir_path"
    parser = OptionParser(usage)
    parser.add_option('-d', '--draw', help='show the images', action='store_true', default=False,
                      dest='draw')

    options, args = parser.parse_args()

    print('options', options)
    print('args', args)

    if len(args) != 2:
        parser.error('incorrect number of arguments')

    dataloaders, dataset_sizes = get_dataloaders(
        study_name=None,
        data_dir=args[1],
        batch_size=30,
        batch_eval_ten=15,
        shuffle=False
    )

    img_path = args[0]
    model = get_pretrained_model('densenet161').to(device)

    if options.draw:
        findTOP5pic(model, dataloaders, img_path)
    else:
        findTOP5addr(model, dataloaders, img_path)


if __name__ == '__main__':
    main()



