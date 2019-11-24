import argparse

import dataset
import graph

dsets = [name for name in dataset.__dict__ if callable(dataset.__dict__[name])]

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', '-d', default = 'MNIST', type = str,
                    choices = dsets, help = 'The dataset: | default(MNIST)')
parser.add_argument('--rate', '-r', default = 0.3, type = float,
                    help = 'The percentage rate of labeled data')
parser.add_argument('--graph', '-g', default = 'grf', type = str,
                    help = 'The semi-supervised algorithm to run. Can be '
                           'one of [grf, llgc]. Default (grf)')
parser.add_argument('--sigma', '-s', default = 0.1, type = float,
                    help = 'The sigma parameter for calculation of weight matrix')
parser.add_argument('--alpha', '-a', default = 0.1, type = float,
                    help = 'The alpha parameter for llgc algorithm')

args = parser.parse_args()

def main():
  x, y, c = dataset.__dict__[args.dataset]()

  nsamples = x.shape[0]

  nlabeled = round(nsamples * args.rate)

  nunlabeled = nsamples - nlabeled

  yl = y[:nlabeled]

  if args.graph == 'grf':
    yu = graph.grf(x, yl, c, args.sigma)
  elif args.graph == 'llgc':
    yu = graph.llgc(x, yl, c, args.sigma, args.alpha)
  else:
    raise ValueError('The graph parameter must be one of [grf, llgc], but ' + args.graph + ' is given')

  return (yu == y[nlabeled:]).sum() / nunlabeled

  

if __name__ == '__main__':
  rate = main()
  print(rate)
