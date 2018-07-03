import common
from common import *
from tqdm import tqdm
from meter import AUCMeterMulti, ConfusionMatrixMeterMulti
from dataset import get_dataloaders, any_dataloader
from model import MURA_Net, MURA_Net_Binary, MURA_Net_AG, get_pretrained_model
from optparse import OptionParser

def evaluate(model, dataloader, phase, draw):
    model.eval()

    aucmeter = AUCMeterMulti()
    aucmeter.add_meter(phase, 'blue', '-')

    confusion = ConfusionMatrixMeterMulti()
    for i, data in enumerate(tqdm(dataloader)):
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
            bs, ncrops, c, h, w = images.size()
            outputs = model(images.view(-1, c, h, w))
            outputs = outputs.view(bs, ncrops, -1).mean(1)
            if isinstance(model, MURA_Net_Binary):
                outputs = outputs[:, 1]
            aucmeter[phase].add(outputs, labels)
        preds = (outputs > 0.5).type(torch.LongTensor).reshape(-1)
        confusion.add(preds, labels, study_name)
    confusion.print()

    print('AUC values')
    aucmeter.print()
    if draw:
        aucmeter.plot()



def main():
    usage = 'usage: %prog [options] data_dir'
    parser = OptionParser(usage)
    parser.add_option('-p', '--phase', help='valid or train, assume the same directory structure with the training set, '
                                            'where there should be a file named <PHASE>.csv in the data directory',
                      action='store', type='string', default='valid', dest='phase')
    parser.add_option('-m', '--model', help='the model to evaluate, one of [densenet161, densenet169, resnet50, vgg19, agnet]',
                      action='store', type='string', default='agnet', dest='model')
    parser.add_option('-s', '--study', help='for evaluating on a specific study',
                      action='store', type='string', default='all', dest='study')
    parser.add_option('-d', '--draw', help='draw roc curves', action='store_true', default=False, dest='draw')
    options, args = parser.parse_args()
    print('options', options)
    print('args', args)
    if len(args) != 1:
        parser.error('incorrect number of arguments')

    data_dir = args[0]
    if not options.model in common.model_names:
        parser.error('Unknown model name')
    if not options.study in common.study_names + ['all']:
        parser.error('Unknown study name')

    study_name = None
    if options.study != 'all':
        study_name = options.study

    dataloader = any_dataloader(
        options.phase,
        study_name=study_name,
        data_dir=args[0],
    )

    model = get_pretrained_model(options.model).to(device)

    evaluate(model, dataloader, options.phase, options.draw)

if __name__ == '__main__':
    main()

