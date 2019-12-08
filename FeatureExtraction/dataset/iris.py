import numpy as np

def iris():
  root = '../dataset/iris/iris.data'
  x = []
  y = []

  f = open(root)

  lines = f.readlines()

  for line in lines:
    data, label = line.rsplit(',', 1)
    x.append([float(d) for d in data.split(',')])
    y.append(label)

  x = np.array(x).astype('float32')
  y = np.array(y)

  classes = []
  for c in y:
    if c not in classes:
      classes.append(c)

  classes.sort()

  for i, c in enumerate(classes):
    y[y == c] = i

  y = y.astype('int32')

  f.close()

  return x, y
