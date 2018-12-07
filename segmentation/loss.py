from math import pi

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_yaw_loss(loss_name, **kwargs):
  if loss_name == 'clas8-regr8':
    return Angle360MixedLoss(N=8, weight_yaw_regr=kwargs['weight_yaw_regr'], regr_per_angle=True)
  elif loss_name == 'clas8-regr1':
    return Angle360MixedLoss(N=8, weight_yaw_regr=kwargs['weight_yaw_regr'], regr_per_angle=False)
  elif loss_name == 'clas8':
    return Angle360MixedLoss(N=8, weight_yaw_regr=0, regr_per_angle=False)
  elif loss_name == 'clas12':
    return Angle360MixedLoss(N=12, weight_yaw_regr=0, regr_per_angle=False)
  elif loss_name == 'cos':
    return Angle360CosLoss()
  elif loss_name == 'cos-sin':
    return Angle360CosSinLoss()
  else:
    raise NotImplemented('Yaw loss "%s" is not implemented.' % loss_name)


class Angle360Loss(nn.Module):
  def __init__(self):
    super(Angle360Loss, self).__init__()

  def prediction2angle(self, x):
    raise NotImplemented()

  def metrics(self, inputs, target):
    raise NotImplemented()

  @staticmethod
  def angle360_l1(inputs, targets):
#    print('Metrics: pred:', inputs, 'gt:', targets, 'diff:', out)
      out = torch.abs(inputs - targets)
      out = torch.min(torch.abs(inputs + 360. - targets), out)
      out = torch.min(torch.abs(inputs - 360. - targets), out)
      return out


class Angle360CosSinLoss(Angle360Loss):
  def __init__(self):
    super(Angle360CosSinLoss, self).__init__()
    self.criterion = torch.nn.SmoothL1Loss()

  def prediction2angle(self, x):
    return torch.atan2(x[1][:,0], x[1][:,1])

  def forward(self, inputs, targets):
    targets_sin = torch.sin(targets * pi / 180.)
    targets_cos = torch.cos(targets * pi / 180.)
    targets = torch.stack([targets_sin, targets_cos], dim=1)
    assert isinstance(inputs, tuple), inputs
    return self.criterion(inputs[1], targets)

  def metrics(self, inputs, targets):
    assert isinstance(inputs, tuple), inputs
    angles = self.prediction2angle(inputs)
    return Angle360Loss.angle360_l1(angles, targets)


class Angle360CosLoss(Angle360Loss):

  def prediction2angle(self, x):
    return inputs[1] * 180. / pi

  def forward(self, inputs, targets):
    assert isinstance(inputs, tuple), inputs
    return (1. - torch.cos(inputs[1] - targets * (pi / 180.))).mean()

  def metrics(self, inputs, targets):
    assert isinstance(inputs, tuple), inputs
    return Angle360Loss.angle360_l1(inputs[1] * 180. / pi, targets)


class Angle360MixedLoss(Angle360Loss):

  def angle2prediction(self, x):
    x = x / 360. * self.N
    x_int1 = torch.remainder(torch.floor(x + 0.5), self.N).long()
    x_frac1 = (torch.remainder(x + 0.5, self.N) - 0.5).float() - x_int1.float()
    use_next = (x_frac1 >= 0).long()
    use_prev = 1 - use_next
    x_int2 = torch.remainder((x_int1 - 1) * use_prev + (x_int1 + 1) * use_next, self.N) 
    x_frac2 = (torch.remainder(x + 0.5, self.N) - 0.5).float() - x_int2.float()
    return x_int1, x_frac1, x_int2, x_frac2

  def prediction2angle(self, x):
    assert isinstance(x, tuple), x
    x_int = torch.argmax(x[0], dim=1)
    if self.regr_per_angle:
      x_frac = torch.diagonal(x[1][:, x_int])
    else:
      x_frac = x[1]
    return torch.remainder((x_int.float() + x_frac) * 360. / self.N, 360.)

  def __init__(self, N, weight_yaw_regr, regr_per_angle=False):
    super(Angle360MixedLoss, self).__init__()
    self.N = 8
    self.criterion_int = torch.nn.CrossEntropyLoss()
    self.criterion_frac = torch.nn.SmoothL1Loss()
    self.weight_yaw_regr = weight_yaw_regr
    self.regr_per_angle = regr_per_angle

  def forward(self, inputs, targets):
    targets1_int, targets1_frac, targets2_int, targets2_frac = self.angle2prediction(targets)

    assert isinstance(inputs, tuple), inputs
    loss_int1  = self.criterion_int  (inputs[0], targets1_int)
    loss_int2  = self.criterion_int  (inputs[0], targets2_int)
    if self.regr_per_angle:
        loss_frac1 = self.criterion_frac (torch.diagonal(inputs[1][:,targets1_int]), targets1_frac)
        loss_frac2 = self.criterion_frac (torch.diagonal(inputs[1][:,targets2_int]), targets2_frac)
#            print('Yaw:', targets.cpu().numpy()[0],
#                  'Int:', torch.argmax(inputs[0], dim=1).detach().cpu().numpy()[0],
#                  'vs gt', targets1_int.cpu().numpy()[0],
#                  'Frac: ', torch.diagonal(inputs[1][:,targets1_int]).detach().cpu().numpy()[0],
#                  'vs gt:', targets1_frac.cpu().numpy()[0],
#                  'loss_int1:', loss_int1.detach().cpu().numpy().tolist(),
#                  'loss_frac1:', loss_frac1.detach().cpu().numpy().tolist())
    else:
        loss_frac1 = self.criterion_frac (inputs[1], targets1_frac)
        loss_frac2 = self.criterion_frac (inputs[1], targets2_frac)
    return torch.min(input=(loss_int1 + self.weight_yaw_regr * loss_frac1),
                      other=(loss_int2 + self.weight_yaw_regr * loss_frac2))

  def metrics(self, inputs, targets):
    inputs = self.prediction2angle(inputs).detach()
    return Angle360Loss.angle360_l1(inputs, targets)


# Recommend
class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss2d(weight, size_average)

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs), targets)


class BalanceLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(BalanceLoss2d, self).__init__()
        self.weight = weight

    def forward(self, inputs1, inputs2):
        prob1 = F.softmax(inputs1)[0, :19]
        prob2 = F.softmax(inputs2)[0, :19]
        print prob1
        prob1 = torch.mean(prob1, 0)
        prob2 = torch.mean(prob2, 0)
        print prob1
        entropy_loss = - torch.mean(torch.log(prob1 + 1e-6))
        entropy_loss -= torch.mean(torch.log(prob2 + 1e-6))
        return entropy_loss


class Entropy(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(Entropy, self).__init__()
        self.weight = weight

    def forward(self, inputs1):
        prob1 = F.softmax(inputs1[0, :19])
        entropy_loss = torch.mean(torch.log(prob1))  # torch.mean(torch.mean(torch.log(prob1),1),0
        return entropy_loss

class Diff2d(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(Diff2d, self).__init__()
        self.weight = weight

    def forward(self, inputs1, inputs2):
        return torch.mean(torch.abs(F.softmax(inputs1) - F.softmax(inputs2)))

class Symkl2d(nn.Module):
    def __init__(self, weight=None, n_target_ch=21, size_average=True):
        super(Symkl2d, self).__init__()
        self.weight = weight
        self.size_average = size_average
        self.n_target_ch = 20
    def forward(self, inputs1, inputs2):
        self.prob1 = F.softmax(inputs1)
        self.prob2 = F.softmax(inputs2)
        self.log_prob1 = F.log_softmax(self.prob1)
        self.log_prob2 = F.log_softmax(self.prob2)

        loss = 0.5 * (F.kl_div(self.log_prob1, self.prob2, size_average=self.size_average)
                      + F.kl_div(self.log_prob2, self.prob1, size_average=self.size_average))

        return loss


# this may be unstable sometimes.Notice set the size_average
def CrossEntropy2d(input, target, weight=None, size_average=False):
    # input:(n, c, h, w) target:(n, h, w)
    n, c, h, w = input.size()

    input = input.transpose(1, 2).transpose(2, 3).contiguous()
    input = input[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0].view(-1, c)

    target_mask = target >= 0
    target = target[target_mask]
    loss = F.cross_entropy(input, target, weight=weight, size_average=False)
    if size_average:
        loss /= target_mask.sum().data[0]

    return loss


def get_prob_distance_criterion(criterion_name, n_class=None):
    if criterion_name == 'diff':
        criterion = Diff2d()
    elif criterion_name == "symkl":
        criterion = Symkl2d(n_target_ch=n_class)
    elif criterion_name == "nmlsymkl":
        criterion = Symkl2d(n_target_ch=n_class, size_average=True)
    else:
        raise NotImplementedError()

    return criterion
