from __future__ import division

from .knn import KNN as knn

import numpy as np



class KMeans(object):
  def __init__(self, x, y, k = 1):
    self.k = k
    self._parse_classes(y)
    self._calculate_parameters(x, y, k)

  def _parse_classes(self, y):
    self._classes = []
    for c in y:
      if c not in self._classes:
        self._classes.append(c)
    self._classes.sort()
    self._num_classes = len(self._classes)

  def _calculate_parameters(self, x, y, k):
    nsamples, nfeatures = x.shape
    self._prototypes = np.empty([k * self._num_classes, nfeatures], dtype = np.float32)
    self._protolabels = np.empty(k * self._num_classes, dtype = np.int32)

    for c in self._classes:
      xc = x[y == c, :]
      mc = xc.mean(axis = 0)
      pc = self._prototypes[c * k:(c + 1) * k, :]
      npc = np.empty([k, nfeatures], dtype = np.float32)
      dc = np.empty([k, xc.shape[0]], dtype = np.float32)
      pc[:] = mc + np.random.randn(k, nfeatures).astype('float32')
      self._protolabels[c * k : (c + 1) * k] = c
      max_iter = 100
      min_err = 0.01
      iter = 0
      err = min_err + 1
      while iter < max_iter and err > min_err:
        iter += 1
        # calculate the distance between each sample and prorotype
        for i, p in enumerate(pc):
          _xc = xc - p
          dc[i, :] = (_xc * _xc).sum(axis = 1)

        # assign each sample to its nearest prototype
        c = dc.argmin(axis = 0).astype('int32')

        # calculate the new means for each cluster
        for kid in range(k):
          # print(c == kid)
          npc[kid, :] = xc[c == kid, :].mean(axis = 0)

        # calculate the err between pc and npc
        _pc = npc - pc
        err = (_pc * _pc).sum(axis = 1).max()

        pc[:] = npc
    self._cls = knn(self._prototypes, self._protolabels)

  def __call__(self, x):
    return self._cls(x)
