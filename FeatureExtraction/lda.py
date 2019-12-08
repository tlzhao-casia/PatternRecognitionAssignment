from __future__ import division

import numpy as np

def lda(x, y, k):
  classes = []
  for c in y:
    if c not in classes:
      classes.append(c)

  nclasses = len(classes)
  mu_cls = [None] * nclasses 
  sigma_cls = [None] * nclasses
  p_cls = [None] * nclasses

  nsamples = x.shape[0]
  for c in classes:
    xi = x[y == c, :]
    ni = xi.shape[0]
    mui = xi.mean(axis = 0)
    mi = mui.reshape(-1, 1)
    sigmai = xi.T.dot(xi) / ni - mi.dot(mi.T)

    mu_cls[c] = mui
    sigma_cls[c] = sigmai
    p_cls[c] = ni / nsamples

  sw = np.zeros_like(sigma_cls[0])
  sb = np.zeros_like(sigma_cls[0])

  for c in classes:
    sw += p_cls[c] * sigma_cls[c]

  for i in classes:
    for j in classes:
      mi = mu_cls[i]
      mj = mu_cls[j]
      mij = (mi - mj).reshape(-1, 1)
      sb += p_cls[i] * p_cls[j] * mij.dot(mij.T)

  s, v, d = np.linalg.svd(sw)
  v = 1 / np.sqrt(v)
  whiten = s.dot(np.diag(v))
  
  sb = whiten.T.dot(sb).dot(whiten)

  s, v, d = np.linalg.svd(sb)

  w = d[:k, :].dot(whiten.T)

  return x.dot(w.T), w 
