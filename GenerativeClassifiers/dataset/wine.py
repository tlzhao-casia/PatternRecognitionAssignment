import numpy as np

def wine():
  data = '../dataset/wine/wine.data'
  f = open(data)
  lines = f.readlines()
  f.close()

  x, y = [line.split(',', 1)[1] for line in lines], [line.split(',', 1)[0] for line in lines]

  x = [[float(d) for d in line.split(',')] for line in x]
  y = [int(d) for d in y]

  x = np.array(x)
  y = np.array(y).astype('int')
  y -= 1

  x = x.transpose(1, 0)[2:]

  mean = x.mean(axis = 1)
  x = x - mean.reshape(-1, 1)
  var = np.sqrt((x * x).sum(axis = 1) / x.shape[1])
  x = x / var.reshape(-1, 1)

  sigma =  x.dot(x.T) / x.shape[1]

  s, v, d = np.linalg.svd(sigma)

  x = s[0:4, :].dot(x)

  return x, y
