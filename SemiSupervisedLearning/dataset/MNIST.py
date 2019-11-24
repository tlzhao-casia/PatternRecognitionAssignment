import os
import struct
import numpy as np

def MNIST():
  data_root = '../dataset/mnist'
  train_images = os.path.join(data_root, 'train-images.idx3-ubyte')
  train_labels = os.path.join(data_root, 'train-labels.idx1-ubyte') 
  
  f_images = open(train_images, 'rb')
  header = f_images.read(16)

  images = []
  for i in range(60000):
    data = f_images.read(28 * 28)
    data = [d for d in data]
    images.append(data)
  f_images.close()
  images = np.array(images).astype('float32') / 255

  f_labels = open(train_labels, 'rb')
  header = f_labels.read(8)
  labels = f_labels.read(60000)
  f_labels.close()
  labels = [l for l in labels]
  labels = np.array(labels)

  indices = np.array([i for i in range(60000)])
  selected_indices = np.zeros(2000)
  for c in range(10):
    c_idx = indices[labels == c]
    np.random.shuffle(c_idx)
    selected_indices[c * 200 : c * 200 + 200] = c_idx[:200]

  selected_indices = selected_indices.astype('int')
  np.random.shuffle(selected_indices)
 
  return images[selected_indices, :], labels[selected_indices]
