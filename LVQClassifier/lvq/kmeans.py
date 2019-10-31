import torch
import numpy as np

def kmeans(x, k, iters):
  ret = torch.zeros(k, x.size(1), device = x.device)
  idx = [i for i in range(x.size(0))]
  np.random.shuffle(idx)
  ret.copy_(x[idx[:k],:])

  for t in range(iters):
    _x = [[] for c in range(k)]
    for i in range(x.size(0)):
      xi = x[i,:].flatten()
      d = (ret - xi.view(1,-1)).norm(dim = 1)
      c = d.argsort(descending = False)[0].item()
      _x[c].append(xi)
    for c in range(k):
      xc = torch.stack(_x[c])
      ret[c,:] = xc.mean(dim = 0)
  return ret
