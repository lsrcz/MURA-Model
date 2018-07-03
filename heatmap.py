import torch
from torchvision.datasets.folder import pil_loader
from localize import tsimg2img, crop_heat, add_boundedbox, add_heatmap_ts
from common import device
from optparse import OptionParser
from model import get_pretrained_model
from PIL import Image
from gcam import _preprocess, gcam


def main():
    usage = "usage: %prog [option] img_path"
    parser = OptionParser(usage)
    parser.add_option('-m', '--model', help='choose the model to generate.one of [\'densenet169\',\'densenet161\',\'resnet50\',\'vgg19\',\'agcnn\']', action='store', type='string', default='densenet161',
                      dest='model')

    options, args = parser.parse_args()

    print('options', options)
    print('args', args)

    if len(args) != 1:
        parser.error('incorrect number of arguments')

    path = [args[0]]
    model = get_pretrained_model(options.model).to(device)

    img_pil = list(map(pil_loader, path))
    img_tensor = list(map(_preprocess, img_pil))
    img_variable = torch.stack(img_tensor).to(device)
    p, x = gcam(model, img_variable)
    print(p)
    t = crop_heat(x, img_variable, threshold=0.63)

    img_pil[0].save('./orig.png')
    Image.fromarray(
        add_boundedbox(x[0], add_heatmap_ts(x[0], img_variable[0], need_transpose_color=False), threshold=0.63,
                       need_transpose_color=True)).save('./boundbox.png')
    tsimg2img(t[0]).save('./croped.png')

if __name__ == '__main__':
    main()