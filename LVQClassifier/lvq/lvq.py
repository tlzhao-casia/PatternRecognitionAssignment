import time

import torch
import numpy as np

from scipy.cluster.vq import kmeans

class LVQ(object):
  def __init__(self, dims, num_classes, k):
    self.dims = dims
    self.num_classes = num_classes
    self.k = k
    self.prototype = torch.randn(num_classes * k, dims).float()
    self.label = np.zeros(num_classes * k).astype('int32')

  def init_prototypes(self, x, y, iters):
    print('Initialize prototypes with kmeans...')
    for n in range(self.num_classes):
      print('Class id: %d | %d'%(n, self.num_classes))
      start = time.time()
      indices = (y == n).astype('uint8')
      xn = x[indices, :]
      idx = [i for i in range(xn.size(0))]
      np.random.shuffle(idx)
      xn = xn[idx[:1000]]
      codebook, _ = kmeans(xn, self.k, iters)
      self.prototype[n * self.k:(n + 1) * self.k,:].copy_(torch.from_numpy(codebook))
      self.label[n * self.k:(n + 1) * self.k] = n
      end = time.time()
      print('Runing time: %.2f min'%((end - start) / 60))

  def to(self, device):
    self.prototype.to(device)

  def state_dict(self):
    return {
      'prototype': self.prototype,
      'label': self.label
    }

  def load_state_dict(self, state_dict):
    fields = ['prototype']
    for field in fields:
      assert(field in state_dict)
      self.__dict__[field].copy_(state_dict[field])
    self.label = state_dict['label']

  def __call__(self, x):
    x = x.to(self.prototype.device)
    d = torch.zeros(x.size(0), self.prototype.size(0), device = x.device)
    for i in range(x.size(0)):
      xi = x[i, :]
      _x = self.prototype - xi.view(1, -1)
      d[i,:] = _x.norm(dim = 1).flatten()
    return d 
