import numpy as np
from sklearn.model_selection import train_test_split
import os

# define the number of input units
input = 4

# define source and target dataset size
source_size = 500
target_size = 500

# initialize the constants and target dataset input and output
epsilon = np.random.normal(0, 1, 1)
x_target = np.abs(np.random.uniform(0, 1, target_size * input).reshape((target_size,input)))
w_target = np.random.randn(input,1)
y_target = x_target.dot(w_target) + epsilon


# initialize the variables for sources
num_sources  = 0
x = []
y = []




# Generate the sources with different delta values
for delta in np.random.uniform(-0.5, 0.5, 5):
    print delta
    num_sources += 1
    x_source = np.abs(np.random.uniform(0, 1, source_size * input).reshape((source_size,input)))
    w_source = w_target + delta * np.random.randn(w_target.shape[0], w_target.shape[1])
    y_source = x_source.dot(w_source) + epsilon

    x.append(x_source)
    y.append(y_source)

x = np.array(x)
y = np.array(y)


#get the min and max y values from target and sources
max_y = max(y_target)
min_y = min(y_target)

for index in range(num_sources):
    min_y = min(min(y[index]), min_y)
    max_y  = max(max(y[index]), max_y)

# initialize the min and max desired values of y
a = 0
b = 1

# normalize the y values for target and save the target dataset
y_target = a + (y_target - min_y)*(b - a)/(max_y - min_y)
x_train, x_test, y_train, y_test  = train_test_split(x_target, y_target, test_size=0.95, random_state=45)
traindata = np.c_[x_train, y_train]
testdata = np.c_[x_test, y_test]

directory = './synthetic_data'

if not os.path.isdir(directory):
    os.mkdir(directory)

np.savetxt(directory+'/target_train.csv', traindata, delimiter=',')
np.savetxt(directory+'/target_test.csv', testdata, delimiter=',')


# normalize the source datasets and save the souce data
for index in range(num_sources):
    x_source = x[index]
    y_source = a + (y[index] - min_y)*(b - a)/(max_y - min_y)
    sourcedata = np.c_[x_source, y_source]
    np.savetxt(directory+'/source'+str(index+1)+'.csv', sourcedata, delimiter=',')
