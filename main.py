from common import *
from dataset import get_dataloaders
from model import MURA_Net
from train import train_model
import os

dataloaders, dataset_sizes = get_dataloaders(
    study_name=None,
    data_dir='MURA-v1.0',
    batch_size=50,
    shuffle=True
)

print(dataset_sizes)

model = MURA_Net()
model = model.to(device)

# model.load_state_dict(torch.load('models/model_XR_WRIST.pth'))
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=1, verbose=True)

model = train_model(model, optimizer, dataloaders, scheduler, dataset_sizes, 500)
# torch.save(model.state_dict(), 'models/model_hand_auc.pth')

