#!/usr/bin/env python3.4
# -*- coding: utf-8 -*-

from constants import get_all_emotions, get_all_landmarks
from itertools import islice
import re
import glob
import csv

def get_lm3_features(file_path):
  fi = open(file_path, 'r')
  save_to = None
  lm = get_all_landmarks()
  features = dict(zip(lm, ['NaN'] * len(lm)))
  # every other line has points with the feature label
  for line in fi:
    if save_to is None and line.strip() in features:
      save_to = line.strip()
    elif save_to != None:
      features[save_to] = parse_points(line.strip())
      save_to = None

  return features

def parse_points(line):
  return [float(x) for x in line.split(' ')]

def get_lm3_files_paths():
  '''
  Return all file paths of lm3 files in the data directory
  '''
  file_paths = []
  data_dir = r'data/bosphorusDB/__files__/__others__/**/*.lm3'
  for file_path in glob.glob(data_dir):
    # Only read lm3 files which are related to emotions
    for x in get_all_emotions():
      if x in file_path:
        file_paths.append(file_path)
  return file_paths

def create_csv(features):
  '''
  Create CSV file of features
  '''
  # Create x,y and z coordinate names for each landmark
  headers = ['Label']
  for landmark in get_all_landmarks():
    headers.append(landmark + '-x')
    headers.append(landmark + '-y')
    headers.append(landmark + '-z')


  # Write out to .csv file
  features.insert(0, headers)
  csv_file = r'lm3.csv'
  with open(csv_file, 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=headers)
    writer.writeheader()
    writer.writerows(features)


if __name__ == '__main__':
  # file_paths = get_lm3_files_paths()
  # features = [parse_lm3_features(file_path) for file_path in file_paths]
  # create_csv(features)
  file_path = '../../data/bosphorusDB/__files__/__others__/BosphorusDB_p1/bs000/bs000_E_ANGER_0.lm3'
  print(get_lm3_features(file_path))
