import torch

phases = ['train', 'valid']
study_names = ['XR_ELBOW', 'XR_FINGER', 'XR_FOREARM', 'XR_HAND', 'XR_HUMERUS', 'XR_SHOULDER', 'XR_WRIST']
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cpu_device = torch.device("cpu")

model_pos = {
    'densenet161': 'models/model_dense161.pth',
    'densenet169': 'models/model_dense169.pth',
    'resnet50': 'models/model_resnet.pth',
    'vgg19': 'models/models_vgg.pth',
    'agnet': 'models/model_total.pth'
}
model_names = {
    'densenet161', 'densenet169',
    'resnet50', 'vgg19', 'agnet'
}