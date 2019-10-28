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

  return x.transpose(1, 0), y
