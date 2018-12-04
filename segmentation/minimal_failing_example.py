import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from models import drn


class Model(nn.Module):
  def __init__(self, submodules):
    super(Model, self).__init__()
    self.submodules = nn.ModuleList(submodules)

  def forward(self, x):
    for submodule in self.submodules:
      x = submodule(x)
    return x


model = drn.__dict__.get('drn_d_105')(
    pretrained=None, num_classes=1000, input_ch=3)

#base = nn.Sequential(*list(model.children())[:-2])
base = Model(list(model.children())[:-2])
print ('base parameters:', len(list(base.parameters())))
print ('first base parameter:', list(base.parameters())[0])

optimizer = torch.optim.Adam(base.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()
base.train()

imgs = torch.ones((1,3,64,64), dtype=torch.float32)
lbls = torch.ones((1,8,8), dtype=torch.int64)
imgs, lbls = Variable(imgs), Variable(lbls)

optimizer.zero_grad()
preds = base(imgs)
loss = criterion(preds, lbls)
loss.backward()
optimizer.step()


