from common import *
from dataset import get_dataloaders
from model import MURA_Net_Binary
from train import train_model
import time
study_name = 'XR_HUMERUS'
dataloaders, dataset_sizes = get_dataloaders(
    study_name=study_name,
    data_dir='MURA-v1.0',
    batch_size=50,
    batch_eval_ten=15,
    shuffle=True
)

print(dataset_sizes)

model = MURA_Net_Binary()
model = model.to(device)

# model.load_state_dict(torch.load('models/model_XR_WRIST.pth'))
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=1, verbose=True)

model = train_model(model, optimizer, dataloaders, scheduler, dataset_sizes, 500)
torch.save(model.state_dict(), 'models/model_binary_' + study_name + str(int(time.mktime(time.localtime(time.time())))) + '.pth')

