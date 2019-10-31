import os
import argparse

import numpy as np
import torch
import gaussian
import dataset

dataset_names = [name for name in dataset.__dict__ if not name.startswith('__')]
classifier_names = [name for name in gaussian.__dict__ if not name.startswith('__')]

parser = argparse.ArgumentParser()

# args for dataset
parser.add_argument('--data', '-d', type = str, default = 'iris',
                     choices = dataset_names,
                     help = 'The dataset for training.')
parser.add_argument('--val-rate', type = float, default = 0.2,
                    help = 'The sample rate for validition data.')

# args for classifiers
parser.add_argument('--cls', '-c', type = str, default = 'qdf',
                    help = 'The classifier to use')
parser.add_argument('--gamma', type = float, default = 0.1,
                    help = 'gamma parameter for rda')
parser.add_argument('--beta', type = float, default = 0.1,
                    help = 'beta parameter for rda')
parser.add_argument('-k', type = int, default = 1,
                    help = 'number of engivalues to reserve')
parser.add_argument('--h', type = float, default = 1,
                    help = 'the width of window')
# gpu-ids
parser.add_argument('--gpu-id', type = int, default = 0,
                    help = 'Device(s) to run on')

args = parser.parse_args()

np.random.seed(0)

def main():
  # prepare dataset
  x,y = dataset.__dict__[args.data]()
  num_samples = x.shape[1] 
  num_train = round(num_samples * (1 - args.val_rate))
  num_val = num_samples - num_train
  assert(num_val > 0 and num_train > 0)
  indices = list(range(num_samples))
  np.random.shuffle(indices)
  train_x = x[:,indices[:num_train]]
  train_y = y[indices[:num_train]]
  val_x = x[:,indices[num_train:]]
  val_y = y[indices[num_train:]]

  train_x = torch.from_numpy(train_x).float()#.to(args.gpu_id)
  train_y = torch.from_numpy(train_y).int()#.to(args.gpu_id)
  val_x = torch.from_numpy(val_x).float()#.to(args.gpu_id)
  val_y = torch.from_numpy(val_y).int()#.to(args.gpu_id)

  # create classifier
  if args.cls.lower() == 'qdf':
    cls = gaussian.QDF(train_x, train_y)
  elif args.cls.lower() == 'ldf':
    cls = gaussian.LDF(train_x, train_y)
  elif args.cls.lower() == 'rda':
    cls = gaussian.RDA(train_x, train_y, args.gamma, args.beta)
  elif args.cls.lower() == 'mqdf':
    cls = gaussian.MQDF(train_x, train_y, args.k)
  elif args.cls.lower() == 'parzen':
    cls = gaussian.Parzen(train_x, train_y, args.h)

  test_acc = test(cls, val_x, val_y)

  print('Test acc: %.2f'%(test_acc * 100))

def test(cls, val_x, val_y):
  y = cls(val_x)
  numel = val_y.numel()
  num_acc = (y == val_y.long()).sum().item() 
  return num_acc / numel

if __name__ == '__main__':
  main()
