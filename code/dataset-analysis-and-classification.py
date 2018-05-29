
# coding: utf-8

# # Analysis and Classification of The Bosphorus Database Dataset

# The analysis and classification of the Bosphorus Database Dataset project is to assign one of the six basic emotions (anger, disgust, fear, happy, sadness, and surprise) to a set of labeled facial landmark positions.
# 
# 
# The Bosphorus Database Dataset emotion samples were provided as text or binary files, with either an ``.lm2`` (2D landmark file with the corresponding labels), ``.lm3`` (3D landmark file with the corresponding labels), or ``.btn`` (coordinate file, both 3D coordinates and corresponding 2D image
# coordinates) extension. In this notebook we'll use the ``.lm3`` files, which are parsed by the [lm3_parser.py](../parsers/lm3_parser.py) and transformed to a ``.csv`` file for convenience.

# The Bosphorus Database Dataset emotion samples were provided as text or binary files, with either an ``.lm2`` (2D landmark file with the corresponding labels), ``.lm3`` (3D landmark file with the corresponding labels), or ``.btn`` (coordinate file, both 3D coordinates and corresponding 2D image
# coordinates) extension. In this notebook we'll use the ``.lm3`` files, which are parsed by the [lm3_parser.py](../parsers/lm3_parser.py) and transformed to a ``.csv`` file for convenience.

# ## Load Dataset

# In[1]:


import pandas as pd

data_df = pd.read_csv(r'../data/lm3.csv')


# ## Visualize Dataset Format and Number of Features

# The dataset consists of a total of 453 samples and 78 features (26 landmarks, each in the x, y and z coordinates ``26*3=78``) and a label associative to it (79 features in total).

# In[2]:


print('Total number of samples: \t%s' % data_df.shape[0])
print('Total number of features: \t%s' % len(data_df.columns.values))
data_df.head()


# ## Data Cleansing

# Unfortunately, some samples do not provide the value of a feature(s), which means that the same feature is also undefined across the 3 dimensions. The [lm3_parser.py](../parsers/lm3_parser.py) handles this use-case by simply setting the value of that feature across the 3 dimensions to ``NaN``.

# #### Drop Features Where NaN Is Present

# In[3]:


print('Total number of features before dropping NaN: \t%s' % len(data_df.columns.values))
data_df = data_df.dropna(axis=1, how='any')
print('Total number of features after dropping NaN: \t%s' % len(data_df.columns.values))
data_df.head()


# ## Data Normalization

# Normalization refers to the process of standardizing the values of independent features of a dataset. Since many of the machine learning techniques use distance to compute the difference between two or more distinct samples, a feature within these samples that has a broad range of values will dominate the process. In order to avoid this, the range of all features are normalized so that each feature contributes approximately proportionately to the computation.

# In[4]:


from sklearn.preprocessing import normalize

# Separate features from their labels
features = [x != 'Label' for x in data_df.columns.values]
all_values = data_df.loc[:, features].values
all_labels = data_df.loc[:,['Label']].values

# Scale values (mean = 0 and variance = 1)
all_values =  normalize(all_values, axis=1)


# ## Dimensionality Reduction
# 
# Furthermore, as the number of features of a dataset increases,  the concept of distance becomes less meaningful (which once again, does not help the set of machine learning techniques that rely on distance computations). This phenomenon is known as the curse of dimensionality and can be alleviated by using dimensionality reduction techniques.
# 
# #### Principal Component Analysis (PCA)
# 
# PCA is a statistical method that can be used to reduce the dimensionality of a dataset. It works by first finding the principal component that accounts for as much of the variance in the data features as possible. Next, each succeeding component in turn is selected such that it has the highest variance possible under the restriction that it is orthogonal to the previous principal components.

# In[5]:


# Use PCA to reduce dimensionality
from sklearn.externals import joblib
from sklearn.decomposition import PCA

pca = PCA(n_components = 15, whiten = True)
principal_components = pca.fit_transform(all_values)
principal_components_df = pd.DataFrame(data = principal_components)

# export model
joblib.dump(pca, 'pca-model.pkl') 

# Add the label to the dataframe with the principal components
pca_df = pd.concat([data_df[['Label']], principal_components_df], axis = 1)
pca_df.head()


# Visualize the PCA 2d projection

# In[6]:


import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure(figsize = (5,5))
ax = fig.add_subplot(1,1,1) 
x_label = pca_df.columns.values[1]
y_label = pca_df.columns.values[2]
ax.set_xlabel(x_label, fontsize = 12)
ax.set_ylabel(y_label, fontsize = 12)
ax.set_title('3 Components PCA', fontsize = 15)

labels = np.unique(data_df.loc[:,['Label']].values)
colors = ['b', 'g', 'r', 'c', 'm', 'y']
for label, color in zip(labels, colors):
    indexes_to_keep = pca_df['Label'] == label
    ax.scatter(
        pca_df.loc[indexes_to_keep, x_label],
        pca_df.loc[indexes_to_keep, y_label],
        c = color,
        s = 50
    )
ax.legend(labels)
ax.grid()
plt.show()


# #### t-Distributed Stochastic Neighbor Embedding  (t-SNE)

# t-SNE is another technique that can be used to perform dimensionality reduction. It embeds high-dimensional data in a low-dimensional space by modeling each high-dimensional object by a two- or three-dimensional point such that objects that are alike are modeled by nearby points and dissimilar objects are modeled otherwise with a high probability.

# In[7]:


from sklearn import manifold

tsne = manifold.TSNE(n_components = 15, init = 'pca', random_state = 0, method='exact')
tsne_components = tsne.fit_transform(all_values)
tsne_components_df = pd.DataFrame(data = tsne_components)

# Add the label to the dataframe with the t-SNE components
tsne_df = pd.concat([data_df[['Label']], tsne_components_df], axis = 1)
tsne_df.head()


# Visualize the t-SNE 2d projection

# In[8]:


plt.scatter(tsne_components[:,0], tsne_components[:,1], c=colors, s=50)
plt.show()


# ## Class Imbalance

# Class imbalance refers to the phenomenon where some classes (labels) of a dataset have more samples than others. This is a problem because the machine learning algorithms will tend to focus on the classification of the samples that are overrepresented while ignoring or misclassifying the underrepresented samples.

# In[9]:


rows, cols = tsne_df.shape
print(tsne_df.groupby('Label').count()[0])


# From the output above we can see the dataset has a class imbalance problem. The emotion "HAPPY" has 106 samples, while "SADNESS" has only 66. We'll fix this issue by carefully selecting the number of samples we'll use for training the machine learning algorithms.

# #### Data Split

# We'll use the holdout method to create a training and test set with an ~80/20 split. As mentioned above, the dataset suffers from a class imbalance problem, and to solve this we'll randomly select the same number of samples per class to add to the training set (60 per class, 360 samples = ~80% of the dataset), and the remaining will be added to the test set (no sample is allowed to be in both sets at the same time).

# In[10]:


def get_train_test_split(df):
    '''
    Split data into training and test sets
    '''
    # Shuffle data frame
    df = df.sample(frac=1)
    
    # Select same number of samples per class for train set, remaining go to test set
    num_of_train_inputs = int(rows * 0.8 / 6)
    train_df, test_df = (pd.DataFrame(columns=df.columns.values), pd.DataFrame(columns=df.columns.values))
    labels = np.unique(df.loc[:,['Label']].values)
    for label in labels:
        train_df = train_df.append(df.loc[df['Label'] == label][0:num_of_train_inputs], ignore_index=True)
        test_df = test_df.append(df.loc[df['Label'] == label][num_of_train_inputs:], ignore_index=True)

    # Shuffle data frames (because they were appended in an orderly per label fashion)
    train_df = train_df.sample(frac=1)
    test_df = test_df.sample(frac=1)
    
    # Split train and test datasets into labels/features
    train_values   = train_df.iloc[:,1:].values
    train_labels = train_df.iloc[:,:1].values.ravel()

    test_values   = test_df.iloc[:,1:].values
    test_labels = test_df.iloc[:,:1].values.ravel()
    
    return (train_values, train_labels, test_values, test_labels)


# #### Confusion Matrix

# Before working with the different machine learning methods which we'll use for classification, let's create a helper method which renders a confusion matrix of a specified model prediction output. A confusion matrix is a table often used to analyze the performance of a classifier on samples for which the true values are known (we'll use it to analyze the performance of the machine learning methods in the test set). Each row in the table represents the instances in an actual class while each column instances in a predicted class.

# In[11]:


from sklearn.metrics import confusion_matrix
import itertools

def plot_confusion_matrix(cm, labels):
    '''
    Plot confusion matrix of the specified accuracies and labels
    '''
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(6, 4))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title('Confusion matrix')
    plt.colorbar()
    
    # Draw ticks
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    
    # Normalize
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.2f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.2f}; misclass={:0.2f}'.format(accuracy, misclass))
    plt.show()


# ## Methods

# #### k-Nearest Neighbors (kNN)

# A kNN classifier simply classifies samples by a majority vote of its neighbors, with the object being assigned to the class most common among its ``k`` nearest neighbors.

# Hyper-parameters:
# - ``n_neighbors``: Number of neighbors to use to determine the label of a sample.

# Create a kNN model using the PCA components:

# In[12]:


from sklearn import neighbors
from sklearn.metrics import accuracy_score

# Get train/test values and labels using the PCA components
train_values, train_labels, test_values, test_labels = get_train_test_split(pca_df)

kNNClassifier = neighbors.KNeighborsClassifier(n_neighbors=15).fit(train_values, train_labels)
predicted_labels = kNNClassifier.predict(test_values)
knn_pca_acc = accuracy_score(test_labels, predicted_labels)
knn_pca_cm = confusion_matrix(test_labels, predicted_labels)
plot_confusion_matrix(knn_pca_cm, labels)


# Create a kNN model using the t-SNE components:

# In[13]:


# Get train/test values and labels using the t-SNE components
train_values, train_labels, test_values, test_labels = get_train_test_split(tsne_df)

kNNClassifier = neighbors.KNeighborsClassifier(n_neighbors=15).fit(train_values, train_labels)
predicted_labels = kNNClassifier.predict(test_values)
knn_tsne_acc = accuracy_score(test_labels, predicted_labels)
knn_tsne_cm = confusion_matrix(test_labels, predicted_labels)
plot_confusion_matrix(knn_tsne_cm, labels)


# #### Support Vector Machine (SVM)

# A SVM is a supervised learning algorithm which creates a model that represents the input samples as points in space, in such a way that samples of different classes are divided by a gap that is as wide as possible. New samples are then mapped into that same space and predicted to belong to a class based on which side of the gap they fall.

# Create a SVM model using the PCA components:

# In[14]:


from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC

# Get train/test values and labels using the PCA components
train_values, train_labels, test_values, test_labels = get_train_test_split(pca_df)

clf = LinearSVC()
svm = CalibratedClassifierCV(clf)
svm.fit(train_values, train_labels)

# export model
joblib.dump(svm, 'svm-model.pkl') 

predicted_labels = svm.predict(test_values)
svm_pca_acc = accuracy_score(test_labels, predicted_labels)
svm_pca_cm = confusion_matrix(test_labels, predicted_labels)
plot_confusion_matrix(svm_pca_cm, labels)


# Create a SVM model using the t-SNE components:

# In[15]:


# Get train/test values and labels using the t-SNE components
train_values, train_labels, test_values, test_labels = get_train_test_split(tsne_df)

svm = LinearSVC()
svm.fit(train_values, train_labels)
predicted_labels = svm.predict(test_values)
svm_tsne_acc = accuracy_score(test_labels, predicted_labels)
svm_tsne_cm = confusion_matrix(test_labels, predicted_labels)
plot_confusion_matrix(svm_tsne_cm, labels)


# #### Multilayer Perceptron (MLP)

# A MLP is a feed-forward artificial neural network that uses supervised learning with backpropagation for training the network (update its weights) to essentially map a weighted input to a label.

# Hyper-parameters:
# - ``solver``: The solver for weight optimization (``adam`` is a stochastic gradient-based optimizer).
# - ``activation``: Activation function for the hidden layer, ``f(x) = x``.
# - ``hidden_layer_sizes``: Number of neurons per hidden layer (a single layer with 18 neurons).
# - ``max_iter``: The solver will iterate until convergence or this number of iterations.

# Create a MLP model using the PCA components:

# In[16]:


from sklearn.neural_network import MLPClassifier

# Get train/test values and labels using the PCA components
train_values, train_labels, test_values, test_labels = get_train_test_split(pca_df)

mlp = MLPClassifier(solver='adam', activation='logistic', hidden_layer_sizes=(5), max_iter=2000)
mlp.fit(train_values, train_labels)

# export model
joblib.dump(mlp, 'mlp-model.pkl') 

predicted_labels = mlp.predict(test_values)
mlp_pca_acc = accuracy_score(test_labels, predicted_labels)
mlp_pca_cm = confusion_matrix(test_labels, predicted_labels)
plot_confusion_matrix(mlp_pca_cm, labels)


# Create a MLP model using the t-SNE components:

# In[17]:


# Get train/test values and labels using the t-SNE components
train_values, train_labels, test_values, test_labels = get_train_test_split(tsne_df)

mlp = MLPClassifier(solver='adam', activation='logistic', hidden_layer_sizes=(5,), max_iter=2000)
mlp.fit(train_values, train_labels)
predicted_labels = mlp.predict(test_values)
mlp_tsne_acc = accuracy_score(test_labels, predicted_labels)
mlp_tsne_cm = confusion_matrix(test_labels, predicted_labels)
plot_confusion_matrix(mlp_tsne_cm, labels)


# #### Random Forest

# A random forest is an ensemble learning method for classification that works by creating multiple decision trees during training and predicting the class that is the mode of the classes of the individual decision trees.

# Hyper-parameters:
# - ``n_estimators``: The number of trees in the forest.
# - ``criterion``: The function to measure the quality of a split.
# - ``max_depth``: The maximum depth of the tree.

# Create a Random Forest model using the PCA components:

# In[18]:


from sklearn.ensemble import RandomForestClassifier

# Get train/test values and labels using the PCA components
train_values, train_labels, test_values, test_labels = get_train_test_split(pca_df)

random_forest = RandomForestClassifier(n_estimators=10, criterion='gini', max_depth=6)
random_forest.fit(train_values, train_labels)
predicted_labels = random_forest.predict(test_values)
random_forest_pca_acc = accuracy_score(test_labels, predicted_labels)
random_forest_pca_cm = confusion_matrix(test_labels, predicted_labels)
plot_confusion_matrix(random_forest_pca_cm, labels)


# Create a Random Forest model using the t-SNE components:

# In[19]:


# Get train/test values and labels using the t-SNE components
train_values, train_labels, test_values, test_labels = get_train_test_split(tsne_df)

random_forest = RandomForestClassifier(n_estimators=10, criterion='gini', max_depth=6)
random_forest.fit(train_values, train_labels)
predicted_labels = random_forest.predict(test_values)
random_forest_tsne_acc = accuracy_score(test_labels, predicted_labels)
random_forest_tsne_cm = confusion_matrix(test_labels, predicted_labels)
plot_confusion_matrix(random_forest_tsne_cm, labels)


# ## Conclusion

# In this notebook we have analyzed the Bosphorus Database Dataset, performed data cleansing, normalization, dimensionality reduction (PCA and t-SNE), class imbalance analysis, holdout split, and finally used four different machine learning methods (kNN, SVM, MLP, and Random Forest) to classify the emotions in the dataset. The SVM and MLP models using the principal components of the PCA reduction have the best accuracy, usually around ~70-80%, while the kNN and Random Forest have an accuracy of ~60%.

# In[20]:


pca_models = {
    'kNN with PCA': knn_pca_acc,
    'SVM with PCA': svm_pca_acc,
    'MLP with PCA': mlp_pca_acc,
    'Random Forest with PCA': random_forest_pca_acc
}

for model, acc in pca_models.items():
    print('%s classifier accuracy: \t%.02f%%' % (model, acc))


# In[21]:


tsne_models = {
    'kNN with TSNE': knn_tsne_acc,
    'SVM with TSNE': svm_tsne_acc,
    'MLP with TSNE': mlp_tsne_acc,
    'Random Forest with TSNE': random_forest_tsne_acc
}
    
for model, acc in tsne_models.items():
    print('%s classifier accuracy: \t%.02f%%' % (model, acc))

