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

  return np.array(x).transpose(1, 0), np.array(_y)
