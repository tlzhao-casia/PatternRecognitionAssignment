import numpy as np

def iris(data = '../dataset/iris/iris.data'):
  f = open(data)
  lines = f.readlines()
  f.close()

  lines = [line.rsplit(',', 1)[0] for line in lines]
  lines = [[float(d) for d in line.split(',')] for line in lines]
  x = np.array(lines).transpose(1, 0)
  y = np.zeros(x.shape[1]).astype('int')
  y[0:50] = 0
  y[51:100] = 1
  y[101:150] = 2

  return x, y
