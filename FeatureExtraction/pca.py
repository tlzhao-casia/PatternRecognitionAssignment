import numpy as np

def pca(x, k):
  assert(len(x.shape) == 2)
  assert(k <= x.shape[1])

  nsamples = x.shape[0]

  mu = x.sum(axis = 0) / nsamples

  m = mu.reshape(-1, 1)

  sigma = x.T.dot(x) / nsamples - m.dot(m.T)

  s, v, d = np.linalg.svd(sigma)

  d = d[:k, :]

  return x.dot(d.T), d
