import sklearn.preprocessing as preprocess
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(filename):

    # READ DATA INTO DATAFRAME
    df = pd.read_csv(filename, sep=";")

    # GET NUMERICAL AND CATEGORICAL COLUMNS
    cols_numerical = list(df.columns[(df.dtypes=="int")|(df.dtypes=="float")])
    cols_categorical = list(df.columns[df.dtypes=="object"].drop('y'))

    # OUTPUT LABELS (SUCCESS/FAILURE)
    y = pd.get_dummies(df["y"]).values.astype("float32")

    # NUMERICAL DATA
    X = df[cols_numerical]
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    
    # CATEGORICAL DATA
    for name in cols_categorical:
        X = pd.concat((X, pd.get_dummies(df[name])), axis=1)
    
    # CHANGE TO FP32
    X = X.values.astype("float32")

    y = np.reshape(y, (-1,2))
    
    # SPLIT INTO TRAIN AND TEST DATA
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # CONCATENATE FEATURES AND LABELS
    train_data = np.concatenate([X_train, y_train], axis=1)
    test_data = np.concatenate([X_test, y_test], axis=1)

    return train_data, test_data


if __name__ == '__main__':

    # GET TRAIN AND TEST DATA
    train_data, test_data = load_data("bank.csv")
    print("Train Data:{}, Test Data:{}".format(train_data.shape, test_data.shape))
    
    # WRITE DATA TO FILE
    np.savetxt('train.csv', train_data, delimiter=',')
    np.savetxt('test.csv', test_data, delimiter=',')
    