import numpy as np

def grf(x, y, num_classes, sigma):
  w = _calculate_weights(x, sigma)
  d = np.diag(w.sum(axis = 0))

  nsamples = x.shape[0]
  nlabeled = y.shape[0]

  xl = x[:nlabeled]
  xu = x[nlabeled:]
  fl = np.zeros((nlabeled, num_classes))
  for n in range(nlabeled):
    fl[n, y[n]] = 1

  duu = d[nlabeled:,nlabeled:]
  wuu = w[nlabeled:,nlabeled:]
  wul = w[nlabeled:,:nlabeled]
  
  fu = np.linalg.inv(duu - wuu).dot(wul).dot(fl)

  return fu.argmax(axis = 1).astype('int')

def llgc(x, y, num_classes, sigma, alpha):
  w  = _calculate_weights(x, sigma)
  d = np.sqrt(w.sum(axis = 0)).reshape((1,-1))
  w = w / (d.T.dot(d))
  
  nsamples = x.shape[0]
  nlabeled = y.shape[0]
 
  f = np.zeros((nsamples, num_classes))

  for n in range(nlabeled):
    f[n, y[n]] = 1

  f = np.linalg.inv(np.eye(nsamples) - alpha / (1 + alpha) * w).dot(f)

  return f.argmax(axis = 1)[nlabeled:]
  

def _calculate_weights(x, sigma):
  nsamples = x.shape[0]
  w = np.ones((nsamples, nsamples)).astype('float32')
  for i in range(nsamples):
    xi = x[i,:].reshape(1, -1)
    _x = x - xi
    _x = (_x * _x).sum(axis = 1)
    w[i, :] = np.exp(- _x / (2 * sigma * sigma))
    w[i, i] = 0

  return w
  
