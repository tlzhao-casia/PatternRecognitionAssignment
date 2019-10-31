import numpy as np

def wdbc():
  data = '../dataset/wdbc/wdbc.data'
  f = open(data)
  lines = f.readlines()
  f.close()

  x, y = [line.split(',', 2)[2] for line in lines], [line.split(',', 2)[1] for line in lines]
  x = [[float(d) for d in line.split(',')] for line in x]
  x, y = np.array(x), np.array(y)
  _y = np.zeros(len(y)).astype('int')
  _y[y == 'M'] = 0
  _y[y == 'B'] = 1

  x = x.transpose(1,0)

  mean = x.mean(axis = 1)

  x = x - mean.reshape(-1, 1)
  var = np.sqrt((x * x).sum(axis = 1) / x.shape[1])
  x = x / var.reshape(-1, 1)

  sigma = x.dot(x.T) / x.shape[1]

  s, v, d = np.linalg.svd(sigma)

  x = s[0:4, :].dot(x)

  y = np.array(_y)

  return x, y
