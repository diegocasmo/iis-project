'''
This is a collection of methods derived from other python notebooks and placed here for brevity.
'''

import math

import pandas as pd

from constants import get_all_emotions, get_all_landmarks
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize

def get_euclidean_distance(x1, x2, y1, y2, z1, z2):
    '''
    Return the euclidean distance between two points
    '''
    term_1 = x1 - x2
    term_2 = y1 - y2
    term_3 = z1 - z2
    return math.sqrt(term_1 ** 2 + term_2 ** 2 + term_3 ** 2)

def variety(lbl):
    return [lbl+'-x', lbl+'-y', lbl+'-z']

def preprocess_data(dataframe):
    '''
    Remove static features, compute nosetip distance, normalize
    '''
    return dataframe.dropna(axis=1, how='any')


def reduce_features(dataframe, num_of_columns=15):
    '''
    Run some dimensionality reductions on a dataframe
    '''
    features = [x != 'Label' for x in dataframe.columns.values]

    # Separating out the features
    values = dataframe.loc[:, features].values

    # Separating out the label
    labels = dataframe.loc[:,['Label']].values

    # Scale values (mean = 0 and variance = 1)
    values =  normalize(values, axis=1)
    
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
            