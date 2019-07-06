import numpy as np
import time
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable

class Network(object):
    """
    Basic Neural Network Class with 1 hidden Layer and Sigmoid Activation function
    
    Attributes
    ----------
    topology: list
        List representing network topology [INPUT, HIDDEN, OUTPUT]
    train_data: ndarray, shape: (n,m)
        Training Data
    test_data: ndarray, shape: (n,m)
        Test/Validation Data
    lrate: float
        Learning rate for Stochastic Gradient Descent
    W1: ndarray
        Weights for input-hidden layers
    B1: ndarray
        Biases for input-hidden layers
    W2: ndarray
        Weights for hidden-output layers
    B2: ndarray
        Biases for input-hidden layers
    """
    def __init__(self, topology, train_data, test_data, learn_rate = 0.5):
        
        # INITIALIZE NETWORK ATTRIBUTES
        self.topology = topology
        self.train_data = train_data
        self.test_data = test_data
        self.lrate = learn_rate
        self.s_size=topology[0]
        self.h_size=topology[1]
        self.a_size=topology[2]
        self.fc1 = nn.Linear(self.s_size, self.h_size)
        self.fc2 = nn.Linear(self.h_size, self.a_size)

        # ASSIGN RANDOM SEED
        np.random.seed(int(time.time()))

        # WEIGHTS AND BIASES
        self.W1 = np.random.randn(self.topology[0], self.topology[1]) / np.sqrt(self.topology[0])
        self.B1 = np.random.randn(1, self.topology[1]) / np.sqrt(self.topology[1])
        self.W2 = np.random.randn(self.topology[1], self.topology[2]) / np.sqrt(self.topology[1])
        self.B2 = np.random.randn(1, self.topology[2]) / np.sqrt(self.topology[1])

        # PLACEHOLDER FOR OUTPUT OF HIDDEN LAYER AND OUTPUT LAYER
        self.hidout = np.zeros((1, self.topology[1]))
        self.out = np.zeros((1, self.topology[2]))

    @staticmethod
    def sigmoid(x):
        """
        Sigmoid Function
        """
        # x = x.astype(np.float128)
        return 1 / (1 + np.exp(-x))

    def sample_er(self, actualout):
        """
        Sample Error Function

        Parameters
        ----------
        actualout: ndarray (n, out)
            Actual Expected output
        
        Returns
        -------
        sqerror: float
            Mean square Error
        """
        error = np.subtract(self.out, actualout)
        sqerror = np.sum(np.square(error)) / self.topology[2]
        return sqerror

    @staticmethod
    def calculate_rmse(observed, targets):
        """
        Calculate Root Mean Squared Error

        Parameters
        ----------
        observed: ndarray (n, m)
            Observed output of network
        targets: ndarray (n, m)
            Expected Target output of network
        
        Returns
        -------
        rmse: float
            Root Mean square Error
        """
        
        rmse = np.sqrt((np.square(np.subtract(observed, targets))).mean())
#        print(rmse)
        return rmse

    # def forward_pass(self, x):
    #     """
    #     Networ Forward Pass
    #     """
        # OUTPUT OF HIDDEN LAYER
        # z1 = X.dot(self.W1) - self.B1
        # self.hidout = self.sigmoid(z1)
        # # OUTPUT OF THE LAST LAYER
        # z2 = self.hidout.dot(self.W2) - self.B2
        # self.out = self.sigmoid(z2)
        
       

    
    def decode(self, weights):
        """
        Reshape Weights and Biases to 2D Arrays
        """
        # SIZE OF INPUT-HIDDEN AND HIDDEN-OUTPUT WEIGHTS
        # 
        # w_layer1_size = self.topology[0] * self.topology[1]
        # w_layer2_size = self.topology[1] * self.topology[2]
        
        # # INPUT-HIDDEN WEIGHTS
        # w_layer1 = w[0:w_layer1_size]
        # self.W1 = np.reshape(w_layer1, (self.topology[0], self.topology[1]))
        
        # # HIDDEN-OUTPUT WEIGHTS
        # w_layer2 = w[w_layer1_size: w_layer1_size + w_layer2_size]
        # self.W2 = np.reshape(w_layer2, (self.topology[1], self.topology[2]))
        
        # # BIASES 
        # self.B1 = w[w_layer1_size + w_layer2_size :w_layer1_size + w_layer2_size + self.topology[1]]
        # self.B2 = w[w_layer1_size + w_layer2_size + self.topology[1] :w_layer1_size + w_layer2_size + self.topology[1] + self.topology[2]]
        print("right now at the decode and the wts are",weights)
        s_size = self.s_size
        h_size = self.h_size
        a_size = self.a_size
		
        # separate the weights for each layer
        fc1_end = (s_size*h_size)+h_size
        fc1_W = torch.from_numpy(weights[:s_size*h_size].reshape(s_size, h_size))
        fc1_b = torch.from_numpy(weights[s_size*h_size:fc1_end])
        fc2_W = torch.from_numpy(weights[fc1_end:fc1_end+(h_size*a_size)].reshape(h_size, a_size))
        fc2_b = torch.from_numpy(weights[fc1_end+(h_size*a_size):])
        # set the weights for each layer
        self.fc1.weight.data.copy_(fc1_W.view_as(self.fc1.weight.data))
		# print(self.fc1.weight.data)
        self.fc1.bias.data.copy_(fc1_b.view_as(self.fc1.bias.data))
        self.fc2.weight.data.copy_(fc2_W.view_as(self.fc2.weight.data))
        self.fc2.bias.data.copy_(fc2_b.view_as(self.fc2.bias.data))
        
# =============================================================================
#         print("fc 1 wt layer",self.fc1.weight.data)
#         print("fc 1 bias layer",self.fc1.bias.data)
#         print("fc 2 wt layer",self.fc2.weight.data)
#         print("fc 1 bias layer",self.fc1.bias.data)
# 
# =============================================================================

    # def encode(self):
    #     """
    #     Reshape weights to 1D Array
    #     """
        
    #     # RESHAPE TO 1D ARRAY
    #     w1 = self.W1.ravel()
    #     w2 = self.W2.ravel()

    #     # CONCATENATE WEIGHTS
    #     w = np.concatenate([w1, w2, self.B1, self.B2])
    #     return w

    @staticmethod
    def softmax(fx):
        """
        Softmax Function
        """
        ex = np.exp(fx)
        sum_ex = np.sum(ex, axis = 1)
        sum_ex = np.multiply(np.ones(ex.shape), sum_ex[:, np.newaxis])
        probability = np.divide(ex, sum_ex)
        return probability

    @staticmethod
    def calculate_accuracy(actual, targets):
        """
        Accuracy Calculation Function
        """
        # ARGMAX
        
        out = np.argmax(targets, axis=1)
        # TARGET
        y_out = np.argmax(actual, axis=1)
        count = 0
        # MATCH
        for index in range(y_out.shape[0]):
            if out[index] == y_out[index]:
                count += 1
        # CALCULATE ACCURACY
        accuracy = float(count)/y_out.shape[0] * 100
        return accuracy

    def generate_output(self, data, w):
        """
        Generate output for given data and weights
        """
        # GENERATE AND UPDATE WEIGHT MATRICES
        print("rightnow at the genereate output in network.py, the wts are",w)
        self.decode(w)

        # INIT VARIABLES
        # size = data.shape[0]
        # Input = np.zeros((1, self.topology[0]))
        # fx = np.zeros((size,self.topology[2]))
        
        # # READ DATA ROW BY ROW AND CARRY OUT FORWARD PASS
        # for i in range(0, size):
        #     Input = data[i, 0:self.topology[0]]
        #     self.forward_pass(Input)
        #     fx[i] = self.out
        train=data[:,0:self.topology[0]]
        train=Variable(torch.from_numpy(train)).float()
        # print(train.shape)
# =============================================================================
#         print("fc 1 wt layer",self.fc1.weight.data)
#         print("fc 1 bias layer",self.fc1.bias.data)
#         print("fc 2 wt layer",self.fc2.weight.data)
#         print("fc 1 bias layer",self.fc1.bias.data)
# =============================================================================

        x = F.relu(self.fc1(train))
        x = F.sigmoid(self.fc2(x))
        return x.detach().numpy()

    def evaluate_fitness(self, w):
        """
        Fitness Function

        Parameters
        ----------
        w: ndarray (n,)
            1D representation of weights
        
        Returns
        -------
        rmse: float
            Root Mean Squared Error on training data
            
        """
        print("right now at evaluate firness in network.py, the wts here are",w)
        data = self.train_data
        y = data[:, self.topology[0]: self.topology[0] + self.topology[2]]
        fx = self.generate_output(data, w)
        return self.calculate_rmse(fx, y)