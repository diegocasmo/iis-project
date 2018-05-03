from struct import *
import numpy as np

def read_int16(f):
  return unpack('h', f.read(2))[0] # short (int16)

def read_int32(f):
  return unpack('I', f.read(4))[0] # int (uint32)

def read_float64(f):
  return unpack('d', f.read(8))[0] # double (float64)

def read_chars(f, length):
  data = []
  inc = 0
  for b in iter_unpack('s', f.read(length)):
    data.append(b[0].decode('ascii'))
    inc += 1;
  return ''.join(data)

def read_float64s(f, nrows):
  data = []
  # read nrows of float64 bytes at a time
  for b in iter_unpack('d' * nrows, f.read()):
    data.append(list(b))

  # transpose the array
  return np.array(data).T.tolist()

def read_bntfile(fname):
  f = open(fname, 'rb')
  points = [];
  try:
    nrows = read_int16(f)
    ncols = read_int16(f)
    zmin = read_float64(f)

    fname_length = read_int16(f)

    imfile = read_chars(f, fname_length)

    data_length = read_int32(f)
    points = read_float64s(f, nrows*ncols)
  finally:
    f.close()
  #     225 * 185 = 41625
  print(nrows, ncols, zmin, fname_length)
  print(imfile)
  print(data_length) # 208125
  print(points[90:100])

if __name__ == "__main__":
  fn = '../data/bosphorusDB/__files__/__others__/' + \
       'BosphorusDB_p1/bs000/bs000_E_ANGER_0.bnt'
  read_bntfile(fn);