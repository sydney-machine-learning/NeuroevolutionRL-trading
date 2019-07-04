import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_data(path):

    # READ DATA FROM CSV
    df = pd.read_csv(path, header=None)
    
    # EXTRACT LABELS AND FEATURES
    labels = df.loc[:, 16]
    features = df.loc[:, list(range(0,16))]

    # ONE-HOT ENCODE LABELS
    labels = pd.get_dummies(labels, prefix='label')
    
    # NORMALIZE FEATURES
    for col in features.columns:
        _mean = features.loc[:, col].mean()
        _std = features.loc[:, col].std()
        features.at[:, col] = (df.loc[:, col]-_mean)/_std
    
    # CONVERT TO NUMPY ARRAY
    features = features.values
    labels = labels.values
    
    # SPLIT INTO TRAIN AND TEST DATA
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.25, random_state=42)

    # CONCATENATE FEATURES AND LABELS
    train_data = np.concatenate([X_train, y_train], axis=1)
    test_data = np.concatenate([X_test, y_test], axis=1)

    return train_data, test_data

    

if __name__ == '__main__':
    filename = 'pen-data.csv'
    train_data, test_data = load_data(filename)
    print("train_data shape: {}, test_data shape: {}".format(train_data.shape, test_data.shape))
    np.savetxt('train.csv', train_data, delimiter=',')
    np.savetxt('test.csv', test_data, delimiter=',')