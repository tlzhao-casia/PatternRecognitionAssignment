import os
import time
import argparse

import torch
import numpy as np
from dataset import olhwdb
import pca
from lvq import kmeans
import lvq

parser = argparse.ArgumentParser()

# args for data set
parser.add_argument('--data-root', '-d', default = '../dataset/OLHWDB1.1',
                    help = 'The root directory of OLHWDB1.1 dataset')
parser.add_argument('--dims', default = None, type = int,
                    help = 'The dimension of PCA compressed data')

# args for lvq
parser.add_argument('--k', default = 1, type = int,
                    help = 'Number of prototypes for each class')

# args for kmeans
parser.add_argument('--kmeans-iter', default = 10000, type = int,
                    help = 'Number of iterations for kmeans initialization')

# gpu-id
parser.add_argument('--gpu-id', default = 0, type = int,
                    help = 'Device to run the algorithm on')

args = parser.parse_args()

def main():
  # prepare the dataset
  print('Parsing training set...')
  train_dir = os.path.join(args.data_root, 'OLHWDB1.1trn')
  train_set = olhwdb.OLHWDB(train_dir)

  print('Parsing test set...')
  test_dir = os.path.join(args.data_root, 'OLHWDB1.1tst')
  test_set = olhwdb.OLHWDB(test_dir)

  train_x = torch.from_numpy(train_set.x).to(args.gpu_id)
  train_y = train_set.y

  test_x = torch.from_numpy(test_set.x).to(args.gpu_id)
  test_y = test_set.y

  # Compress the data with PCA if args.dims is set
  if args.dims is not None:
    assert(args.dims < train_x.size(1))
    print('Compressing dataset to %d dims...'%(args.dims))
    train_x, sigma, mean = pca.pca(train_x, args.dims)
    test_x = test_x - mean.view(1, -1)
    test_x = test_x.matmul(sigma)

  # create lvq model
  net = lvq.LVQ(train_x.size(1), train_set.num_classes, args.k)
  net.to(args.gpu_id)

  # init lvq prototypes with kmeans
  net.init_prototypes(train_x, train_y, args.kmeans_iter)

  test_acc = test(net, test_x, test_y)

  print('Test acc for k-means initialization: %.2f'%(test_acc))

def test(net, test_x, test_y):
  d = net(test_x)
  idx = d.argmin(dim = 1).flatten()
  n = test_x.size(0)
  test_y = torch.from_numpy(test_y).to(idx.device)
  num_acc = (idx == test_y.long()).sum().item()
  return num_acc / n
 

if __name__ == '__main__':
  main()
