import torch

class LVQ(object):
  def __init__(self, dims, num_classes, k):
    self.prototype = torch.randn(num_classes * k, dims).float()
    self.label = torch.zeros(num_classes * k).int()

  def to(self, device):
    self.prototype.to(device)
    self.label.to(device)

  def state_dict(self):
    return {
      'prototype': self.prototype,
      'label': self.label
    }

  def load_state_dict(self, state_dict):
    fields = ['prototype', 'label']
    for field in fields:
      assert(field in state_dict)
      self.__dict__[field].copy_(state_dict[field])

  def __call__(self, x):
    x = x.to(self.prototype.device)
    d = torch.zeros(x.size(0), self.prototype.size(0), device = x.device)
    for i in range(x.size(0)):
      xi = x[i, :]
      _x = self.prototype - xi.view(1, -1)
      d[i,:] = (_x * _x).sum(dim = 1).flatten()
    return d 
