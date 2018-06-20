import torch

phases = ['train', 'valid']
study_names = ['XR_ELBOW', 'XR_FINGER', 'XR_FOREARM', 'XR_HAND', 'XR_HUMERUS', 'XR_SHOULDER', 'XR_WRIST']
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")