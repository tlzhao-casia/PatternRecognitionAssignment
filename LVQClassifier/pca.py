def pca(x, k):
  mean = x.mean(dim = 0)
  _x = x - mean.view(1, -1)
  sigma = _x.transpose(1,  0).matmul(_x) / _x.size(0)
  s, v, d = sigma.svd()

  sigma = sigma[:,:k]
  _x = _x.matmul(sigma)
  
  return _x, sigma, mean
