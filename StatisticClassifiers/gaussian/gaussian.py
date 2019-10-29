import math

import torch

from .utils import classes
from .utils import inverse

class Gaussian(object):
  def __init__(self, x, y):
    self.x = x
    self.y = y
    self.classes = classes(y)

  def _discriminant_func(self, x, y):
    raise NotImplementedError

  def __call__(self, x):
    y = []
    for c in self.classes:
      y.append(self._discriminant_func(x, c).view(1, -1))
    y = torch.cat(y, dim = 0)
    return y.argmax(dim = 0)
    

class Parsen(Gaussian):
  def __init__(self, x, y, h):
    super(Parsen, self).__init__(x, y)
    self.h = h

  def _func(self, x):
    return (-(x * x).sum(dim = 0)).exp()

  def _discriminant_func(self, x, y):
    xi = self.x[:,self.y == y]
    ret = torch.zeros(x.size(1), device = xi.device)
    for i in range(xi.size(1)):
      _x = (x - xi[:,i].view(-1,1)) / self.h
      ret += self._func(_x)
    return ret * xi.size(1) / self.y.numel()
    
     

class QDF(Gaussian):
  def __init__(self, x, y):
    super(QDF, self).__init__(x, y)
    self.sigma = []
    self.mean = []
    self.log_sigma = []
    self._calculate_sigma_and_mean()

  def _calculate_sigma_and_mean(self):
    for y in self.classes:
      indices = (self.y == y)
      xi = self.x[:,indices]
      mean = xi.mean(dim = 1).flatten()
      xi = xi - mean.view(-1, 1)
      sigma = xi.matmul(xi.transpose(1, 0)) / xi.size(1)
      log_sigma = sigma.det().log()
      self.sigma.append(inverse(sigma))
      self.mean.append(mean)
      self.log_sigma.append(log_sigma)

  def _discriminant_func(self, x, y):
    sigma = self.sigma[y]
    mean = self.mean[y]
    log_sigma = self.log_sigma[y]
    x = x - mean.view(-1, 1)
    ret = torch.empty(x.size(1), device = x.device)
    for i in range(x.size(1)):
      ret[i] = (-x[:,i].view(1,-1).matmul(sigma).matmul(x[:,i].view(-1,1)) - log_sigma).flatten()
    return ret

class RDA(QDF):
  def __init__(self, x, y, gamma, beta):
    super(QDF, self).__init__(x, y)
    self.gamma = gamma
    self.beta = beta
    self.sigma = []
    self.mean = []
    self.log_sigma = []
    self._calculate_sigma_mean()

  def _calculate_sigma_mean(self):
    dim = self.x.size(0)
    sigma = torch.zeros(dim, dim, device = self.x.device)
    for c in self.classes:
      xi = self.x[:,self.y == c]
      meani = xi.mean(dim = 1)
      _xi = xi - meani.view(-1,1)
      sigmai = _xi.matmul(_xi.transpose(1,0)) / _xi.size(1)
      pi = xi.size(1) / self.x.size(1)
      sigma += pi * sigmai
      self.sigma.append(sigmai)
      self.mean.append(meani)

    for i, s in enumerate(self.sigma):
      _s = (1 - self.gamma) * ((1 - self.beta) * s + self.beta * sigma) + self.gamma * s.trace() / dim
      self.sigma[i] = inverse(_s)
      self.log_sigma.append(_s.det().log())

class MQDF(QDF):
  def __init__(self, x, y, k):
    super(QDF, self).__init__(x, y)
    self.k = k
    self.sigma = []
    self.mean = []
    self.log_sigma = []
    self._calculate_sigma_mean()

  def _calculate_sigma_mean(self):
    dim = self.x.size(0)
    for c in self.classes:
      xi = self.x[:,self.y == c]
      meani = xi.mean(dim = 1)
      _xi = xi - meani.view(-1, 1)
      sigmai = _xi.matmul(_xi.transpose(1, 0)) / _xi.size(1)
      s, v, d = sigmai.svd()
      v[self.k:] = v[self.k-1]
      sigmai = s.matmul(torch.diag(v)).matmul(d.transpose(1,0))
      self.sigma.append(inverse(sigmai))
      self.mean.append(meani)
      self.log_sigma.append(sigmai.det().log())

class LDF(Gaussian):
  def __init__(self, x, y):
    super(LDF, self).__init__(x, y)
    self.weight = []
    self.bias = []
    self.log_p = []
    self._calculate_prior_prob()
    self._calculate_weight_bias()

  def _calculate_prior_prob(self):
    for c in self.classes:
      self.log_p.append((self.y == c).sum() / self.x.size(1))

  def _calculate_weight_bias(self):
    mean = self.x.mean(dim = 1)
    x = self.x - mean.view(-1, 1)
    sigma = inverse(x.matmul(x.transpose(1, 0)) / self.x.size(1))
    for c in self.classes:
      indices = (self.y == c)
      xi = self.x[:,indices]
      meani = xi.mean(dim = 1)
      self.weight.append(2 * sigma.matmul(meani.view(-1,1))) 
      self.bias.append(2 * self.log_p[c] - meani.view(1,-1).matmul(sigma).matmul(meani.view(-1,1)))

  def _discriminant_func(self, x, y):
    w = self.weight[y]
    b = self.bias[y]
    return (w.view(1,-1).matmul(x) + b).flatten()
