import numpy as np
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
import csv

import time

def getdata(file, input):
    data = []
    with open(file, 'rb') as f:
        reader = csv.reader(f)
        data = list(reader)
        data = data[:-1]

    for index in range(len(data)):
        clas = data[index][-1]
        if clas == 'Iris-setosa':
            data[index][-1] = 1
            data[index].extend([0, 0])
        elif clas == 'Iris-versicolor':
            data[index][-1] = 0
            data[index].extend([1, 0])
        else:
            data[index][-1] = 0
            data[index].extend([0, 1])

        data[index] = list(map(float, data[index]))

    data = np.array(data)
    x = data[:, :input]
    y = data[:, input:]
    # print(x.shape, y.shape)


    sc_X = StandardScaler()
    x1 = sc_X.fit_transform(x)
    x = normalize(x1, norm='l2')

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.30, random_state = int(time.time()))

    traindata = np.c_[X_train,y_train]
    testdata = np.c_[X_test, y_test]

    return [traindata, testdata]


if __name__ == '__main__':
    data = 'iris'
    train,test = getdata(data+'.csv', input=4)
    np.savetxt(data+'-train.csv', train, delimiter = ',')
    np.savetxt(data+'-test.csv', test, delimiter = ',')