import numpy as np
from constants import get_all_emotions
from sklearn.decomposition import PCA
from sklearn.externals import joblib
from sklearn.preprocessing import normalize

import pandas as pd

# Sample data, do not ship
data_df = pd.read_csv(r'../../data/lm3.csv')
data_df = data_df.dropna(axis=1, how='any')
features = [x != 'Label' for x in data_df.columns.values]
all_values = data_df.loc[:, features].values
r = all_values[0]
print(r)

# Normalize
all_values = normalize([r], axis=1)
print(all_values)

# Load PCA model and transform input
pca = joblib.load('pca-model.pkl')
principal_components = pca.transform(all_values)
print(principal_components)

# Load MLP model and predict probability
mlp = joblib.load('mlp-model.pkl')
predicted_labels = mlp.predict_proba(principal_components)[0]

# Assign each score to its label
confidence = {}
labels = get_all_emotions()
for index,emotion in enumerate(labels):
    confidence[emotion] = predicted_labels[index]

print(confidence)

