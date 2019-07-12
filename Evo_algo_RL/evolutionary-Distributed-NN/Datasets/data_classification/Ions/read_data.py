import numpy as np
from sklearn.model_selection import train_test_split

with open('ionosphere.data', 'r') as file:
    data  = file.readlines()
    processed = np.zeros((len(data), 36))
    i = 0
    for point in data:
        point = point.strip().split(',')
        category = point[-1]
        point = list(map(float, point[:-1]))
        #print(category)
        if category == 'g':
            point.extend([1,0])
        elif category == 'b':
            point.extend([0,1])
        else:
            print("Category:", category)
            raise(ValueError('Incorrect value detected for category!'))
        processed[i] = point
        #print(processed[i])
        i += 1


X = processed[:, :34]
y = processed[:, 34:]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
#print(X_train.shape, y_train.shape)
train_data = np.concatenate([X_train, y_train], axis=1)
test_data = np.concatenate([X_test, y_test], axis=1)

np.savetxt('train.csv', train_data, delimiter=',')
np.savetxt('test.csv', test_data, delimiter=',')


