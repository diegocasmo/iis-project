#!/usr/bin/env python3.4
# -*- coding: utf-8 -*-

from constants import get_all_emotions, get_all_landmarks
from itertools import islice
import glob
import csv
import pdb

def get_defined_landmarks(file_path):
  '''
  Retrieve landmarks defined in lm2 file
  '''
  defined_landmarks = []
  with open(file_path) as file:
    # Start reading file after the 'Labels:' line
    start_line = 5
    for line in islice(file, start_line, None):
      # Stop reading lines as soon as an empty line is discovered
      if line in ['\n', '\r\n']:
        break
      defined_landmarks.append(line.rstrip('\n\r')) # Remove end of line character
  return defined_landmarks

def get_undefined_landmarks_idx(all_landmarks, defined_landmarks):
  '''
  Return index of landmarks which are undefined in the lm2 file
  '''
  undefined_landmarks_idx = []
  for idx, landmark in enumerate(all_landmarks):
    if not landmark in defined_landmarks:
      undefined_landmarks_idx.append(idx)
  return undefined_landmarks_idx

def get_lm2_features(file_path):
  '''
  Return value of each landmark in file
  '''
  try:
    # Find undefined landmark values in lm2 file

    all_landmarks = get_all_landmarks()
    defined_landmarks = get_defined_landmarks(file_path)
    undefined_landmarks_idx = get_undefined_landmarks_idx(all_landmarks, defined_landmarks)

    # Find landmark values defined in lm2 file
    defined_landmarks_values = []
    with open(file_path) as file:
      # Start reading file after the '2D Image coordinates' line
      start_line = len(defined_landmarks) + 7
      for line in islice(file, start_line, None):
        defined_landmarks_values.append(line.rstrip('\n\r'))

    # Fill-in undefined values in landmark as 'NaN'
    for idx in undefined_landmarks_idx:
      defined_landmarks_values.insert(idx, 'NaN')

    # Must have the same number of landmark values as there are possible landmarks
    assert(len(defined_landmarks_values) == len(all_landmarks))

    # Unpack feature values in x and y coordinates
    features = []
    for value in defined_landmarks_values:
      if value == 'NaN':
        features += ['NaN'] * 2 # Feature value is undefined in both coordinates
      else:
        features += map(float, value.split())

    # Must have twice the number of features now
    assert(len(features) == 2 * len(all_landmarks))

    return features
  except Exception as e:
    print('Error: ' + file_path)


def get_lm2_label(file_path):
  '''
  Given a file path, return the label its data represents
  '''
  for x in get_all_emotions():
    if x in file_path:
      return x

def parse_lm2_features(file_path):
  '''
  Return the label and features represented by an lm2 file
  '''
  return [get_lm2_label(file_path)] + get_lm2_features(file_path)

def get_lm2_files_paths():
  '''
  Return all file paths of lm2 files in the data directory
  '''
  file_paths = []
  data_dir = r'data/bosphorusDB/__files__/__others__/**/*.lm2'
  for file_path in glob.glob(data_dir):
    # Only read lm2 files which are related to emotions
    for x in get_all_emotions():
      if x in file_path:
        file_paths.append(file_path)
  return file_paths

def create_csv(features):
  '''
  Create CSV file of features
  '''
  # Create x and y coordinate names for each landmark
  headers = ['Label']
  for landmark in get_all_landmarks():
    headers.append(landmark + '-x')
    headers.append(landmark + '-y')

  # Write out to .csv file
  features.insert(0, headers)
  csv_file = r'data/lm2.csv'
  with open(csv_file, 'w') as output:
    writer = csv.writer(output, lineterminator='\n')
    writer.writerows(features)

if __name__ == '__main__':
  file_paths = get_lm2_files_paths()
  features = [parse_lm2_features(file_path) for file_path in file_paths]
  create_csv(features)
