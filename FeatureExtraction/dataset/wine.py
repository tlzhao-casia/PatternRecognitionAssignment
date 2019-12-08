import numpy as np

def wine():
  root = '../dataset/wine/wine.data'
  
  x = []
  y = []

  f = open(root)
  lines = f.readlines()

  for line in lines:
    label, data = line.split(',', 1)
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

  return x, y
