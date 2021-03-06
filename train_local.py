import time
import copy
import torch
from common import *
from tqdm import tqdm
import torch.nn.functional as F

from dataset import get_dataloaders
from gcam import gcam
from localize import crop_heat
from meter import AUCMeterMulti, ConfusionMatrixMeterMulti
from model import MURA_Net_AG


def train_local(model, optimizer, dataloaders, scheduler, dataset_sizes, num_epochs, name='model'):
    assert isinstance(model, MURA_Net_AG)
    model.eval()
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_idx = -1
    costs = {x:[] for x in phases + ['valid_study']}
    accs = {x:[] for x in phases + ['valid_study']}
    print('Train batches:', len(dataloaders['train']))
    print('Valid batches:', len(dataloaders['valid']), '\n')
    for epoch in range(num_epochs):
        aucmeter = AUCMeterMulti()
        aucmeter.add_meter('valid', 'red', '-')
        aucmeter.add_meter('valid_study', 'blue', '-')
        if epoch > best_idx + 10:
            print("The accuracy didn't improved in 10 epoches")
            break
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        for phase in phases + ['valid_study']:
            model.global_net.eval()
            if phase == 'train':
                model.local_net.train()
            else:
                model.local_net.eval()
            confusion = ConfusionMatrixMeterMulti()
            running_loss = 0.0
            for i, data in enumerate(tqdm(dataloaders[phase])):
                images = data['image']['norm']
                labels = data['label'].type(torch.FloatTensor)
                metadatas = data['metadata']
                weights = metadatas['wt'].type(torch.FloatTensor)
                study_name = metadatas['study_name']
                sizes = metadatas['dataset_size'].numpy()

                images = images.to(device)
                labels = labels.to(device)
                weights = weights.to(device)

                optimizer.zero_grad()

                if phase == 'valid_study':
                    bs, ncrops, c, h, w = images.size()
                    images_view = images.view(-1, c, h, w)
                    _, cam = gcam(model.global_net, images_view)
                    newimgs = crop_heat(cam, images_view).to(device)
                else:
                    _, cam = gcam(model.global_net, images)
                    newimgs = crop_heat(cam, images).to(device)
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model.local_net(newimgs)
                    if phase == 'valid_study':
                        outputs = outputs.view(bs, ncrops, -1).mean(1)

                    loss = F.binary_cross_entropy(outputs, labels, weight=weights)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    else:
                        aucmeter[phase].add(outputs, labels)
                running_loss += loss.item() * images.size(0)
                preds = (outputs > 0.5).type(torch.LongTensor).reshape(-1)


                confusion.add(preds, labels, study_name)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = confusion.accuracy()

            if phase == 'valid_study':
                print('valid_study:')
            else:
                print('{} Loss: {:.4f}'.format(phase, epoch_loss))
            confusion.print()

            costs[phase].append(epoch_loss)
            accs[phase].append(epoch_acc)

            if phase == 'valid':
                scheduler.step(epoch_loss)
            if phase == 'valid_study':
                temp_auc ,_, _ = aucmeter.meters['valid_study'].value()
                if temp_auc > best_acc:
                    best_idx = epoch
                    best_acc = temp_auc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(model.state_dict(),
                               'models/' + name + '_' + str(epoch) + '_' +
                               str(temp_auc) + '_' +
                               str(int(time.mktime(time.localtime(time.time()))))
                               + '.pth')

        # aucmeter.plot()
        aucmeter.print()
        time_elapsed = time.time() - since
        print('Time elapsed: {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60
        ))
        print()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60
    ))
    print('Best valid Acc: {:4f}'.format(best_acc))
    model.load_state_dict(best_model_wts)
    return model


def main():
    dataloaders, dataset_sizes = get_dataloaders(
        study_name = None,
        data_dir='MURA-v1.0',
        batch_size=25,
        batch_eval_ten=15,
        shuffle=True
    )

    print(dataset_sizes)
    model = MURA_Net_AG('densenet169')
    model = model.to(device)

    model.load_global_dict(torch.load('models/model.pth'))
    model.load_local_dict(torch.load('models/model.pth'))

    # model.load_state_dict(torch.load('models/model_XR_WRIST.pth'))
    optimizer = torch.optim.Adam(model.local_net.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=1, verbose=True)

    model = train_local(model, optimizer, dataloaders, scheduler, dataset_sizes, 500)
    torch.save(model.state_dict(), 'models/model_local_169_bi.pth')

if __name__ == '__main__':
    main()