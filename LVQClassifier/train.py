import os
import time
import argparse

import torch
import numpy as np

import olhwdb
import pca

parser = argparse.ArgumentParser()

# args for data set
parser.add_argument('--data-root', '-d', default = '../../dataset/OLHWDB1.1',
                    help = 'The root directory of OLHWDB1.1 dataset')
parser.add_argument('--dims', default = None, type = int,
                    help = 'The dimension of PCA compressed data')

# gpu-id
parser.add_argument('--gpu-id', default = 0, type = int,
                    help = 'Device to run the algorithm on')

args = parser.parse_args()

def main():
  # prepare the dataset
  train_dir = os.path.join(args.data_root, 'OLHWDB1.1trn')
  train_set = olhwdb.OLHWDB(train_dir)

  test_dir = os.path.join(args.data_root, 'OLHWDB1.1tst')
  test_set = olhwdb.OLHWDB(test_dir)

  train_x = torch.from_numpy(train_set.x).to(args.gpu_id)
  train_x = torch.from_numpy(train_sey.y).to(args.gpu_id)

  test_x = torch.from_numpy(test_set.x).to(args.gpu_id)
  test_y = torch.from_numpy(test_set.y).to(args.gpu_id)

  # Compress the data with PCA if args.dims is set
  if args.dims is not None:
    assert(args.dims < train_x.size(1))
    train_x, sigma, mean = pca.pca(train_x)
    test_x = test_x - mean.view(1, -1)
    test_x = test_x.matmul(sigma)
  

if __name__ == '__main__':
  main()
