import torch

class LVQOptimizer(object):
  def __init__(self, prototype, label, lr):
    self.prototype = prototype
    self.label = label
    self.lr = lr

  def step(self, x, label_x, d):
    raise NotImplementedError
