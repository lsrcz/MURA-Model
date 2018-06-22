from common import *
from dataset import get_dataloaders
from model import MURA_Net
from train import train_model
import os

dataloaders, dataset_sizes = get_dataloaders(
    study_name=None,
    data_dir='MURA-v1.0',
    batch_size=8,
    shuffle=True
)

print(dataset_sizes)

class Loss(torch.nn.modules.Module):
    def __init__(self, wt1, wt0):
        super(Loss, self).__init__()
        self.wt1 = wt1
        self.wt0 = wt0

    def forward(self, input, target, phase):
        loss = - (self.wt1[phase] * target * input.log() +
                  self.wt0[phase] * (1 - target) * (1 - input).log())
        return loss

model = MURA_Net()
model = model.to(device)


optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=1, verbose=True)

model = train_model(model, optimizer, dataloaders, scheduler, dataset_sizes, 500)
torch.save(model.state_dict(), 'models/model.pth')

