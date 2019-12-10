from __future__ import division
import numpy as np

class KNN(object):
  def __init__(self, x, y, k = 1):
    self._num_classes, self._classes = self._parse_classes(y)
    self.x = x
    self.y = y
    self.k = k

  def _parse_classes(self, y):
    classes = []
    for c in y:
      if c not in classes:
        classes.append(c)
    classes.sort()

    return len(classes), classes

  def __call__(self, x):
    nsamples = x.shape[0]
    score = np.empty([nsamples, self._num_classes], dtype = np.float32)
    for s, data in zip(score, x):
      _x = self.x - data
      d = (_x * _x).sum(axis = 1)
      cls = self.y[d.argsort()[:self.k]].astype('int32')
      for c in self._classes:
        s[c] = (cls == c).sum()

    return score.argmax(axis = 1).astype('int32')
