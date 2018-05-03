#!/usr/bin/env python3.4
# -*- coding: utf-8 -*-

from numpy import genfromtxt
import pdb

def get_numpy_arr_from_csv(file_path):
  '''
  '''
  try:
    return genfromtxt(file_path, delimiter=',')
  except Exception as e:
    print('Error: Unable to read file: %s' % file_path)

if __name__ == '__main__':
  file_path = r'data/lm2.csv'
  data = get_numpy_arr_from_csv(file_path)
  print(data)
