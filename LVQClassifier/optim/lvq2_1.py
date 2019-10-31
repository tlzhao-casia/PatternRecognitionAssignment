from .lvq_optimizer import LVQOptimizer

class LVQ2_1(LVQOptimizer):
  def __init__(self, prototype, label, lr, w):
    super(LVQ2_1, self).__init__(prototype, label, lr)
    self.w = w

  def step(self, x, label_x, d):
    x = x.flatten()
    d = d.flatten()
    idx = d.argsort(descending = False)[:2]

    di = d[idx[0]].item()
    dj = d[idx[1]].item()
    mi = self.prototype[idx[0],:].flatten()
    mj = self.prototype[idx[1],:].flatten()
    label_i = self.label[idx[0]]
    label_j = self.label[idx[1]]
    
    if label_i != label_j and di / dj > (1 - self.w) / (1 + self.w):
      if label_i == label_x:
        mi.sub_(self.lr, mi - x)
        mj.add_(self.lr, mj - x)
      elif label_j == label_x:
        mi.add_(self.lr, mi - x)
        mj.sub_(self.lr, mj - x)
      else:
        if label_i == label_x:
          mi.sub_(self.lr, mi - x)
        else:
          mi.add_(self.lr, mi - x)
    else:
      if label_i == label_x:
        mi.sub_(self.lr, mi - x)
      else:
        mi.add_(self.lr, mi - x)

def lvq2_1(prototype, label, lr, w):
  return LVQ2_1(prototype, label, lr, w) 
