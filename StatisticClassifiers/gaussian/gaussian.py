import torch

from .utils import classes

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
    return y.argmax(dim = 1)
    
     

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
      self.sigma.append(sigma.inverse())
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
    sigma = x.matmul(x.transpose(1, 0)).inverse()
    for c in self.classes:
      indices = (self.y == y)
      xi = self.x[:,indices]
      meani = xi.mean(dim = 1)
      self.weight.append(2 * sigma.matmul(meani.view(-1,1))) 
      self.bias.append(2 * self.log_p[c] - meani.view(1,-1).matmul(sigma).matmul(meani.view(-1,1)))
