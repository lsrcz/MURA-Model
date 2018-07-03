import time
from common import *
from tqdm import tqdm
import torch.nn.functional as F
from meter import AUCMeterMulti, ConfusionMatrixMeterMulti
from dataset import get_dataloaders
from model import MURA_Net, MURA_Net_Binary, MURA_Net_AG
from train import train_model
import os

def valid(model, dataloaders):
    model.eval()
    print('Valid batches:', len(dataloaders['valid']), '\n')

    aucmeter = AUCMeterMulti()
    aucmeter.add_meter('valid', 'red', '-')
    aucmeter.add_meter('valid_study', 'blue', '-')

    for phase in ['valid', 'valid_study']:
        confusion = ConfusionMatrixMeterMulti()
        for i, data in enumerate(tqdm(dataloaders[phase])):
            images = data['image']['norm']
            labels = data['label'].type(torch.FloatTensor)
            metadatas = data['metadata']
            weights = metadatas['wt']
            study_name = metadatas['study_name']
            sizes = metadatas['dataset_size'].numpy()

            images = images.to(device)
            labels = labels.to(device)
            weights = weights.to(device)

            with torch.no_grad():
                if phase == 'valid':
                    outputs = model(images)
                    if isinstance(model, MURA_Net_Binary):
                        outputs = outputs[:, 1]
                    loss = F.binary_cross_entropy(outputs, labels, weight=weights)
                    aucmeter[phase].add(outputs, labels)
                elif phase == 'valid_study':
                    bs, ncrops, c, h, w = images.size()
                    outputs = model(images.view(-1, c, h, w))
                    outputs = outputs.view(bs, ncrops, -1).mean(1)
                    if isinstance(model, MURA_Net_Binary):
                        outputs = outputs[:, 1]
                    aucmeter['valid_study'].add(outputs, labels)
                else:
                    assert False
            preds = (outputs > 0.5).type(torch.LongTensor).reshape(-1)
            confusion.add(preds, labels, study_name)
        confusion.print()

    # aucmeter.plot()
    print('AUC values')
    aucmeter.print()


def main():
    dataloaders, dataset_sizes = get_dataloaders(
        study_name=None,
        data_dir='MURA-v1.0',
        batch_size=20,
        batch_eval_ten=15,
        shuffle=True
    )
    '''
    model = MURA_Net('densenet161')
    model = model.to(device)
    model.load_state_dict(torch.load('models/model_densenet161_auc.pth'))
    '''
    model = MURA_Net_AG('densenet161')
    model = model.to(device)
    model.load_state_dict(torch.load('models/model_total_2_0.9113079154854713_1530563526.pth'))

    valid(model, dataloaders)

if __name__ == '__main__':
    main()

