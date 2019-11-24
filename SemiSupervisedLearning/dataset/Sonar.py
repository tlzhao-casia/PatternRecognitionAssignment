import numpy as np

def sonar():
  root = '../dataset/sonar/sonar.all-data'
  f = open(root, 'rt')
  lines = f.readlines()
  x = []
  l = []

  labels = []
  label_map = dict()

  for line in lines:
    data, label = line.rsplit(',', 1)
    label = label.strip()
    if label not in labels:
      labels.append(label)
      label_map.update({
        label : len(labels) - 1
      })
    x.append([float(d) for d in data.split(',')])
    l.append(label_map[label])

  x = np.array(x)
  l = np.array(l)

  indices = [i for i in range(len(l))]
  np.random.shuffle(indices)
  x = x[indices, :]
  l = l[indices]

  return x, l, len(labels)
