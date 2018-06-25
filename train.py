import time
import copy
import torch
from common import *
from tqdm import tqdm
import torch.nn.functional as F
from meter import AUCMeterMulti

def train_model(model, optimizer, dataloaders, scheduler, dataset_sizes, num_epochs):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_idx = -1
    costs = {x:[] for x in phases + ['valid_tencrop']}
    accs = {x:[] for x in phases + ['valid_tencrop']}
    print('Train batches:', len(dataloaders['train']))
    print('Valid batches:', len(dataloaders['valid']), '\n')
    for epoch in range(num_epochs):
        aucmeter = AUCMeterMulti()
        aucmeter.add_meter('valid', 'red', '-')
        aucmeter.add_meter('valid_tencrop', 'green', '-')
        if epoch > best_idx + 10:
            print("The accuracy didn't improved in 10 epoches")
            break
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)
        for phase in phases + ['valid_tencrop']:
            if phase == 'train':
                model.train(phase=='train')
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0
            for i, data in enumerate(tqdm(dataloaders[phase])):
                images = data['image']
                labels = data['label'].type(torch.FloatTensor)
                weights = data['metadata']['wt']

                images = images.to(device)
                labels = labels.to(device)
                weights = weights.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    if phase == 'valid_tencrop':
                        bs, ncrops, c, h, w = images.size()
                        outputs = model(images.view(-1, c, h, w))
                        outputs = outputs.view(bs, ncrops, -1).mean(1)
                    else:
                        outputs = model(images)

                    #loss = criterion(outputs, labels, phase).sum()
                    loss = F.binary_cross_entropy(outputs, labels, weight=weights)
                    running_loss += loss
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    else:
                        aucmeter[phase].add(outputs, labels)
                running_loss += loss.item() * images.size(0)
                preds = (outputs > 0.5).type(torch.LongTensor)
                running_corrects += torch.sum(preds.transpose(0,1) == data['label'].data)

            epoch_loss = running_loss / dataset_sizes[phase]
            #print(running_corrects)
            #print(dataset_sizes[phase])
            epoch_acc = running_corrects.type(torch.FloatTensor) / dataset_sizes[phase]
            #print(epoch_acc.dtype)
            #print(running_corrects.dtype)
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc
            ))
            costs[phase].append(epoch_loss)
            accs[phase].append(epoch_acc)
            if phase == 'valid_tencrop':
                scheduler.step(epoch_loss)
                if epoch_acc > best_acc:
                    best_idx = epoch
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
            # if phase != 'train':
            #     aucmeter[phase].plot()
        aucmeter.plot()
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
