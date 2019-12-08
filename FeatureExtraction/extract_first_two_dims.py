import dataset
import pca
import lda

def extract_first_two_dims(d):
  x, y = dataset.__dict__[d]()

  nsamples, nfeatures = x.shape
 
  # original features
  f = open(d + '_ori.feature', 'wt')
  for n in range(nsamples):
    f.write(str(x[n, 0]) + ',' + str(x[n, 1]) + ',' + str(y[n]) + '\n')
  f.close()

  # pca features
  pca_x, pca_w = pca.pca(x, 2)
  f = open(d + '_pca.feature', 'wt')
  for n in range(nsamples):
    f.write(str(pca_x[n, 0]) + ',' + str(pca_x[n, 1]) + ',' + str(y[n]) + '\n')
  f.close()

  # lda features
  lda_x, lda_w = lda.lda(x, y, 2)
  f = open(d + '_lda.feature', 'wt')
  for n in range(nsamples):
    f.write(str(lda_x[n, 0]) + ',' + str(lda_x[n, 1]) + ',' + str(y[n]) + '\n')
  f.close()

  

dset = ['iris', 'wine', 'vowel']

for d in dset:
  extract_first_two_dims(d)
