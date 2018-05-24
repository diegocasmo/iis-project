'''
This is a collection of methods derived from other python notebooks for splitting the test and train
'''
import pandas as pd

from constants import get_all_emotions

def get_train_test_split(df, column_names,index, verbose=False):
    '''
    Split data into training and test sets
    '''
    # Shuffle data frame
    df = df.sample(frac=1)
    # Select same num per class, remaining go to test set
    rows, _ = df.shape
    num_of_inputs = int(rows * 0.8 / 6) # this was formerly the magic number 63.
    cols = ['Label'] +  column_names
    train_df, test_df = (pd.DataFrame(columns=cols), pd.DataFrame(columns=cols))
    for x in get_all_emotions() :
        train_df = train_df.append(df.loc[df['Label'] == x][0:num_of_inputs], ignore_index=True)
        test_df = test_df.append(df.loc[df['Label'] == x][num_of_inputs:], ignore_index=True)

    # Shuffle data frames
    train_df = train_df.sample(frac=1)
    test_df = test_df.sample(frac=1)

        # Just print once
    if verbose and index == 0:
        # Take a look at the labels distribution
        print('--------------------Training--------------------')
        rows, cols = train_df.shape
        print(train_df.groupby('Label').count())
        print('Total number of inputs: %s' % rows)

        print('--------------------Testing--------------------')
        rows, cols = test_df.shape
        print(test_df.groupby('Label').count())
        print('Total number of inputs: %s' % rows)

    # Split train and test labels/data
    train_data   = train_df.iloc[:,1:].values
    train_labels = train_df.iloc[:,:1].values.ravel()

    test_data   = test_df.iloc[:,1:].values
    test_labels = test_df.iloc[:,:1].values.ravel()

    return(train_data,train_labels,test_data,test_labels)
