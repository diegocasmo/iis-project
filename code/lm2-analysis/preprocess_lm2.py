'''
This is a collection of methods derived from other python notebooks and placed here for brevity.
'''

import math

import pandas as pd

from constants import get_all_emotions, get_all_landmarks
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize

def get_euclidean_distance(x1, x2, y1, y2):
    '''
    Return the euclidean distance between two points
    '''
    term_1 = x1 - x2
    term_2 = y1 - y2
    return math.sqrt(term_1 ** 2 + term_2 ** 2)

def variety(lbl):
    return [lbl+'-x', lbl+'-y']

def preprocess_data(dataframe):
    '''
    Remove static features, compute nosetip distance, normalize
    '''
    # print('Num of features:\t%s' % len(data_df.columns.values))

    # Drop features potentiall unuseful features 
    drops = variety('Right nose peak') + variety('Left nose peak') + \
            variety('Left temple') + variety('Right temple') + \
            variety('Right ear lobe') + variety('Left ear lobe') + \
            variety('Nose saddle left') + variety('Nose saddle right') + variety('Chin middle')
    # errors = 'ignore' because it will throw if columns aren't there
    data_df = dataframe.drop(drops, axis=1, errors='ignore')
#     print('after drop un-used\t%s' % len(data_df.columns.values))

    # Drop features where 'NaN' is present
    data_df = data_df.dropna(axis=1, how='any')
#     print('after drop NaN\t\t%s' % len(data_df.columns.values))
#     data_df.head()

    # Compute the distance from all points to the nose tip (intuition: nose tip position is likely to be similar regardless of emotion)

    # Create new dataset using distance from each landmark to the nose tip as features
    all_landmarks = get_all_landmarks()
    valid_landmarks = []

    # Remove landmarks which were previously removed due to undefined values ('NaN')
    columns = data_df.columns.values
    for landmark in all_landmarks:
        if landmark + '-x' in columns and landmark + '-y' in columns:
            valid_landmarks.append(landmark)

    # Create dictionary to temporarily hold distance values
    d = {}
    for landmark in valid_landmarks:
        d[landmark + '-distance'] = []

    # For each feature, compute distance from nose tip
    for _, row in data_df.iterrows():
        for landmark in valid_landmarks:
            nose_x, nose_y = (row['Nose tip-x'], row['Nose tip-y'])
            landmark_x, landmark_y = (row[landmark + '-x'], row[landmark + '-y'])
            distance = get_euclidean_distance(landmark_x, nose_x, landmark_y, nose_y)
            d[landmark + '-distance'].append(distance)

    # Create new dataframe (with distance from each feature to nose tip)
    return pd.concat([data_df[['Label']], pd.DataFrame.from_dict(d)], axis = 1)

def reduce_features(dataframe, num_of_columns=14):
    '''
    Run some dimensionality reductions on a dataframe
    '''
    features = [x != 'Label' for x in dataframe.columns.values]

    # Separating out the features
    values = dataframe.loc[:, features].values

    # Separating out the label
    labels = dataframe.loc[:,['Label']].values

    # Scale values (mean = 0 and variance = 1)
    values =  normalize(values, axis=0)
    
    # Set the number of columns we want to reduce
    column_names = ['Col %s' % (i + 1) for i in range(num_of_columns)]   
    pca = PCA(n_components=num_of_columns, whiten = True)
    principal_components = pca.fit_transform(values)
    principal_df = pd.DataFrame(data = principal_components, columns = column_names)
    # Add the label to the dataframe with the principal components
    principal_labels_df = pd.concat([dataframe[['Label']], principal_df], axis = 1)
#     principal_labels_df.head()

#     tsne = TSNE(n_components=num_of_columns, init='pca', random_state=0, method='exact')
#     tsne_components = tsne.fit_transform(values)
#     tsne_df = pd.DataFrame(data = tsne_components, columns = column_names)
    
    return (column_names, principal_labels_df)
            