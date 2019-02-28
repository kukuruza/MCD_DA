import torch.nn as nn
import torch.nn.functional as F
from grad_reversal import grad_reverse
from weight_init import weight_init


class Feature(nn.Module):
    def __init__(self):
        super(Feature, self).__init__()
        self.conv1 = nn.Conv2d(3072, 3072, kernel_size=3, stride=1, padding=1, groups=3072)
        self.bn1 = nn.BatchNorm2d(3072)
        self.conv2 = nn.Conv2d(3072, 3072, kernel_size=3, stride=1, padding=1, groups=3072)
        self.bn2 = nn.BatchNorm2d(3072)
        self.conv3 = nn.Conv2d(3072, 3072, kernel_size=3, stride=1, padding=1, groups=3072)
        self.bn3 = nn.BatchNorm2d(3072)
        #self.fc1 = nn.Linear(8192, 3072)
        weight_init(self.conv1)
        weight_init(self.conv2)
        weight_init(self.conv3)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), stride=2, kernel_size=2, padding=0)
#        print (x.size())
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), stride=2, kernel_size=2, padding=0)
#        print (x.size())
        x = F.max_pool2d(F.relu(self.bn3(self.conv3(x))), stride=2, kernel_size=2, padding=0)
#        print (x.size())
        x = F.dropout(x, training=self.training)
        x = x.view(x.size(0), 3072)
        return x


class Predictor(nn.Module):
    def __init__(self, clas_out_ch, regr_out_ch):
        super(Predictor, self).__init__()
        self.fc2 = nn.Linear(3072, 2048)
        self.bn2_fc = nn.BatchNorm1d(2048)
        self.fc3 = nn.Linear(2048, clas_out_ch)
        weight_init(self.fc2)
        weight_init(self.fc3)

        self.regr_fc2 = nn.Linear(3072, 2048)
        self.regr_bn2_fc = nn.BatchNorm1d(2048)
        self.regr_fc3 = nn.Linear(2048, regr_out_ch)
        weight_init(self.regr_fc2)
        weight_init(self.regr_fc3)

    def set_lambda(self, lambd):
        self.lambd = lambd

    def forward(self, x, reverse=False):
        if reverse:
            x = grad_reverse(x, self.lambd)
        y = F.relu(self.bn2_fc(self.fc2(x)))
        y = self.fc3(y)

        z = F.relu(self.regr_bn2_fc(self.regr_fc2(x)))
        z = self.regr_fc3(z).squeeze()
        return y, z

