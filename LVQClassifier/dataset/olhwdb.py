import os
import struct

import numpy as np

def find_all_files(root, expand = None):
  ret = []
  for d in os.listdir(root):
    d = os.path.join(root, d)
    if os.path.isdir(d):
      ret += find_all_files(d, expand)
    else:
      if expand is None:
        ret.append(d)
      elif d.endswith(expand):
        ret.append(d)
  return ret

def parse_mpf_file(fname):
  f = open(fname, 'rb')
  # parse the header size, 4B
  h_size = f.read(4)
  h_size = struct.unpack('I', h_size)
  # print('Header size: %d B'%(h_size))

  # parse the format code, 8B
  format_code = f.read(8)
  # print('Format code:', format_code)

  # parse the illustration, (h_size - 62)B
  illustration = f.read(h_size[0] - 62)
  # print('Illustration: ', illustration)

  # parse the code type 20B
  code_type = f.read(20)
  # print('Code type: ', code_type)

  # parse the code length 2B
  code_length = f.read(2)
  code_length = struct.unpack('H', code_length)
  # print('Code length: %s'%(code_length))

  # parse the data type, 20B
  dtype = f.read(20)
  # print('Data type: ', dtype)

  # parse the sample number, 4B
  n_samples = f.read(4)
  n_samples = struct.unpack('I', n_samples)
  # print('Number of samples: %d'%(n_samples))

  # parse the dims, 4B
  dims = f.read(4)
  dims = struct.unpack('I', dims)
  # print('Dimensions: %d'%(dims))

  x = []
  y = []
  for n in range(n_samples[0]):
    code = f.read(code_length[0])
    data = f.read(dims[0])
    x.append([float(d) for d in data])
    y.append(struct.unpack('H', code)[0])
  
  f.close()

  return x, y
  
  
def class_to_idx(y):
  classes = []
  for c in y:
    if c not in classes:
      classes.append(c)
  classes.sort()
  class_idx = {classes[i]:i for i in range(len(classes))}

  for i in range(len(y)):
    y[i] = class_idx[y[i]]

  return class_idx


class OLHWDB(object):
  def __init__(self, root):
    mpfs = find_all_files(root, '.mpf')
    self.root = root
    self.mpfs = mpfs

    fname_x = os.path.join(root, 'data.npy')
    fname_y = os.path.join(root, 'label.npy')
    if os.path.exists(fname_x) and os.path.exists(fname_y):
      self.x = np.load(fname_x)
      self.y = np.load(fname_y)
    else:
      self.x = []
      self.y = []
      for mpf in self.mpfs:
        # print('Parsing mpf file: %s'%(mpf))
        x, y = parse_mpf_file(mpf)
        self.x += x
        self.y += y
      self.x = np.array(x).astype('float32')
      self.y = np.array(y).astype('int32')
      np.save(fname_x, self.x)
      np.save(fname_y, self.y) 
 
    self.class_to_idx = class_to_idx(self.y)
    self.num_classes = len(self.class_to_idx)
  


    # print('Number of classes: %d'%(self.num_classes))    
