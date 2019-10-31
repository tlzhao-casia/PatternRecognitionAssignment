import torch

def classes(y):
  classes = []
  for c in y:
    if c not in classes:
      classes.append(c.item())

  return classes

def inverse(t):
  s, v, d = t.svd()
  mask = (v > 0)
  s = s[:,mask]
  v = 1 / v[mask]
  d = d[:,mask]
  return d.matmul(torch.diag(v)).matmul(s.transpose(1,0))
