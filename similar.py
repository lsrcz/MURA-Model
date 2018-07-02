from common import *
from tqdm import tqdm
import operator
from dataset import get_dataloaders
from model import MURA_Net, MURA_Net_Binary
from CAM import predictWithCAM
import numpy as np
from torchvision import models, transforms
from torchvision.datasets.folder import pil_loader

similarity = {}
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

    img_tensors = find_img_tensor(dataloaders,path)
    img_variable = torch.stack(img_tensors).to(device)
    with torch.no_grad():
        out, cam_ori = predictWithCAM(model, img_variable)
    cam_ori = cam_ori[0]
    for i, data in enumerate(tqdm(dataloaders['valid'])):
        images = data['image']['norm']
        labels = data['label']
        metadatas = data['metadata']
        paths = metadatas['path']

        images = images.to(device)
        with torch.no_grad():
            o, cam = predictWithCAM(model, images)
            for j in range(images.shape[0]):
                similarity[paths[j]] = np.linalg.norm(cam[j] - cam_ori) *  (1 + 1000000 * abs(labels[j] - out[0]))
                '''
                if paths[j] == 'MURA-v1.0/valid/XR_ELBOW/patient11840/study1_positive/image3.png':
                    print(similarity[paths[j]])
                    #print(cam[j])
                    #print(cam_ori)
                '''

    sim = sorted(similarity.items(), key=operator.itemgetter(1), reverse=False)
    for i in range(0,6):
        print(sim[i])




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
    model.load_state_dict(torch.load('models/model161.pth'))

    img_path = 'MURA-v1.0/valid/XR_ELBOW/patient11840/study1_positive/image4.png'
    findTOP5(model, dataloaders, img_path)

if __name__ == '__main__':
    main()

