from __future__ import division

import dataset
import classifiers
from pca import pca
from lda import lda

import numpy as np

def split_dataset(dset):
  x, y = dset()
  nsamples, nfeatures = x.shape
  indices = [i for i in range(nsamples)]
  np.random.shuffle(indices)
  x = x[indices, :]
  y = y[indices]

  ntrain = int(nsamples * 0.8)

  train_x = x[:ntrain, :]
  train_y = y[:ntrain]
  
  test_x = x[ntrain:, :]
  test_y = y[ntrain:]

  return train_x, train_y, test_x, test_y
def val_uci(train_x, train_y, test_x, test_y, fs, dim, cls, f):
  if fs == 'pca':
    train_x, w = pca(train_x, dim)
    test_x = test_x.dot(w.T)
  elif fs == 'lda':
    train_x, w = lda(train_x, train_y, dim)
    test_x = test_x.dot(w.T)

  out = cls(train_x, train_y)(test_x)
  
  acc = (out == test_y).sum() / out.shape[0] * 100
  acc = '%-8.2f'%(acc)

  print(acc)
  f.write(acc)

dsets = ['iris', 'wine', 'vowel']
fss = ['pca', 'lda']
clss = ['LDF', 'QDF', 'KNN']

dims = [
  [(1, 2), (1, 2)],
  [(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), (1, 2)],
  [(1, 2, 3, 4, 5, 6, 7, 8, 9), (1, 2, 3, 4, 5, 6, 7, 8, 9)]
]

for i, d in enumerate(dsets):
  train_x, train_y, test_x, test_y = split_dataset(dataset.__dict__[d])
  for j, fs in enumerate(fss):
    f = open(d + '.' + fs + '.res', 'wt')
    for cls in clss:
      for dim in dims[i][j]:
        val_uci(train_x, train_y, test_x, test_y, fs, dim, classifiers.__dict__[cls], f)
      f.write('\n')
    f.close()
