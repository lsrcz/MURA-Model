import time
import copy
import torch
from common import *
from tqdm import tqdm
import torch.nn.functional as F
from meter import AUCMeterMulti, ConfusionMatrixMeterMulti
from dataset import get_dataloaders
from model import MURA_Net
from train import train_model
import os

dataloaders, dataset_sizes = get_dataloaders(
study_name=None,
data_dir='MURA-v1.0',
batch_size=50,
batch_eval_ten=15,
shuffle=True
)


model = MURA_Net()
model = model.to(device)
model.load_state_dict(torch.load('models/model.pth'))

since = time.time()
print('Valid batches:', len(dataloaders['valid']), '\n')
aucmeter = AUCMeterMulti()
aucmeter.add_meter('valid', 'red', '-')
aucmeter.add_meter('valid_tencrop', 'green', '-')
aucmeter.add_meter('valid_study', 'blue', '-')
confusion = ConfusionMatrixMeterMulti()
model.eval()
running_loss = 0.0
phase = 'valid'
for i, data in enumerate(tqdm(dataloaders['valid'])):
    images = data['image']
    labels = data['label'].type(torch.FloatTensor)
    metadatas = data['metadata']
    weights = metadatas['wt']
    study_name = metadatas['study_name']
    sizes = metadatas['dataset_size'].numpy()

    images = images.to(device)
    labels = labels.to(device)
    weights = weights.to(device)

    outputs = model(images)

    with torch.set_grad_enabled(phase == 'train'):
        outputs = model(images)

        loss = F.binary_cross_entropy(outputs, labels, weight=weights)
        running_loss += loss
        aucmeter[phase].add(outputs, labels)
    running_loss += loss.item() * images.size(0)
    preds = (outputs > 0.5).type(torch.LongTensor).reshape(-1)

    confusion.add(preds, labels, study_name)

epoch_loss = running_loss / dataset_sizes['valid']
epoch_acc = confusion.accuracy()

print('{} Loss: {:.4f} Acc: {:.4f} F1: {:.4f} Kappa: {:.4f}'.format(
    'valid', epoch_loss, confusion.accuracy(), confusion.F1(), confusion.kappa()
))
print('{:>13}{:>7}{:>7}{:>7}'.format('Study', 'Acc', 'F1', 'Kappa'))
for key in confusion.names:
    print('{:>13} {:6.4f} {:6.4f} {:6.4f}'.format(
        key, confusion.accuracy(key), confusion.F1(key), confusion.kappa(key)
    ))

confusion = ConfusionMatrixMeterMulti()
for i, data in enumerate(tqdm(dataloaders['valid_study'])):
    images = data['images']
    labels = data['label'].type(torch.FloatTensor)
    metadatas = data['metadata']
    study_name = metadatas['study_name']
    sizes = metadatas['dataset_size'].numpy()

    images = images.to(device)
    labels = labels.to(device)


    with torch.no_grad():
        bs, ncrops, c, h, w = images.size()
        outputs = model(images.view(-1, c, h, w))
        outputs = outputs.view(bs, ncrops, -1).mean(1)
        aucmeter['valid_study'].add(outputs, labels)

    preds = (outputs > 0.5).type(torch.LongTensor).reshape(-1)
    confusion.add(preds, labels, study_name)

print('{} Acc: {:.4f} F1: {:.4f} Kappa: {:.4f}'.format(
    'valid_study', confusion.accuracy(), confusion.F1(), confusion.kappa()
))
print('{:>13}{:>7}{:>7}{:>7}'.format('Study', 'Acc', 'F1', 'Kappa'))
for key in confusion.names:
    print('{:>13} {:6.4f} {:6.4f} {:6.4f}'.format(
        key, confusion.accuracy(key), confusion.F1(key), confusion.kappa(key)
    ))
