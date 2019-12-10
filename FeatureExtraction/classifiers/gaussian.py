from __future__ import division
import numpy as np

class Gaussian(object):
  def __init__(self, x, y):
    self._classes, self._num_classes = self._parse_classes(y)
    self._calculate_parameters(x, y)
  
  def _parse_classes(self, y):
    classes = []
    for c in y:
      if c not in classes:
        classes.append(c)
    classes.sort()

    return classes, len(classes)

  def _calculate_parameters(self, x, y):
    raise NotImplementedError

  def _calculate_score(self, x, c):
    raise NotImplementedError

  def __call__(self, x):
    nsamples = x.shape[0]
    ret = np.empty([nsamples, self._num_classes], dtype = np.float32)
    
    for c in self._classes:
      ret[:, c] = self._calculate_score(x, c)
    
    return ret.argmax(axis = 1).astype('int32')

class LDF(Gaussian):
  def __init__(self, x, y):
    super(LDF, self).__init__(x, y)

  def _calculate_parameters(self, x, y):
    self._mean = [None for c in self._classes]
    self._p = [None for c in self._classes]
    n = x.shape[0]
    mean = x.mean(axis = 0)
    m = mean.reshape(-1, 1)
    self._sigma = x.T.dot(x) / n - m.dot(m.T)
    self._sigma_inv = np.linalg.pinv(self._sigma)

    for c in self._classes:
      xc = x[y == c, :]
      nc = xc.shape[0]
      self._mean[c] = xc.mean(axis = 0)
      self._p[c] = nc / n
    
  def _calculate_score(self, x, c):
    assert(c >= 0 and c < self._num_classes)
    return x.dot(self._sigma_inv).dot(self._mean[c]) - 0.5 * self._mean[c].dot(self._sigma_inv).dot(self._mean[c]) + np.log(self._p[c])

class QDF(Gaussian):
  def __init__(self, x, y):
    super(QDF, self).__init__(x, y)

  def _calculate_parameters(self, x, y):
    self._mean = [None for c in self._classes]
    self._sigma_inv = [None for c in self._classes]
    self._sigma_det = [None for c in self._classes]

    for c in self._classes:
      xc = x[y == c, :]
      nc = xc.shape[0]
      self._mean[c] = xc.mean(axis = 0)
      mc = self._mean[c].reshape(-1, 1)
      sigmac = xc.T.dot(xc) / nc - mc.dot(mc.T)
      self._sigma_inv[c] = np.linalg.pinv(sigmac)
      self._sigma_det[c] = np.linalg.det(sigmac)

  def _calculate_score(self, x, c):
    _x = x - self._mean[c]

    nsamples = _x.shape[0]

    scores = np.empty(nsamples, dtype = np.float32)

    for n in range(nsamples):
      scores[n] = -_x[n,:].dot(self._sigma_inv[c]).dot(_x[n,:]) - np.log(self._sigma_det[c])

    return scores
