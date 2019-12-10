from __future__ import division

import dataset
import classifiers
import pca
import lda

train = dataset.OLHWDB('../dataset/olhwdb/train')
test = dataset.OLHWDB('../dataset/olhwdb/test')
'''
f = open('olhwdb.kmeans.pca.res', 'wt')

for k in range(1, 11):
  for c in range(1, 4):
    train_x, w = pca.pca(train.x, k * 10)
    test_x = test.x.dot(w.T)
    cls = classifiers.KMeans(train_x, train.y, c)
    out = cls(test_x)
    acc = '%-6.2f'%((out == test.y).sum() / out.shape[0] * 100)
    print(acc)
    f.write(acc)
  f.write('\n')

f.close()
'''

f = open('olhwdb.kmeans.lda.res', 'wt')

for k in range(1, 11):
  for c in range(1, 4):
    train_x, w = lda.lda(train.x, train.y, k * 10)
    test_x = test.x.dot(w.T)
    cls = classifiers.KMeans(train_x, train.y, c)
    out = cls(test_x)
    acc = '%-6.2f'%((out == test.y).sum() / out.shape[0] * 100)
    print(acc)
    f.write(acc)
  f.write('\n')

f.close()
