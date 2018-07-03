import torch
from torchvision.datasets.folder import pil_loader
from heatmap import tsimg2img, crop_heat, add_boundedbox, add_heatmap_ts
from common import device
from optparse import OptionParser
from model import get_pretrained_model
from PIL import Image
from gcam import _preprocess, gcam


def main():
    usage = "usage: %prog [option] img_path"
    parser = OptionParser(usage)

    options, args = parser.parse_args()

    print('options', options)
    print('args', args)

    if len(args) != 1:
        parser.error('incorrect number of arguments')

    path = [args[0]]
    model = get_pretrained_model('densenet161').to(device)

    img_pil = list(map(pil_loader, path))
    img_tensor = list(map(_preprocess, img_pil))
    img_variable = torch.stack(img_tensor).to(device)
    p, x = gcam(model, img_variable)
    t = crop_heat(x, img_variable, threshold=0.63)

    print('Saving images at current directory...')
    print('orig.png ...')
    img_pil[0].save('./orig.png')
    print('boundbox.png ...')
    Image.fromarray(
        add_boundedbox(x[0], add_heatmap_ts(x[0], img_variable[0], need_transpose_color=False), threshold=0.63,
                       need_transpose_color=True)).save('./boundbox.png')
    print('croped.png ...')
    tsimg2img(t[0]).save('./croped.png')

if __name__ == '__main__':
    main()