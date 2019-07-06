import numpy as np
import time

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
        return rmse

    def forward_pass(self, X):
        """
        Networ Forward Pass
        """
        # OUTPUT OF HIDDEN LAYER
        z1 = X.dot(self.W1) - self.B1
        self.hidout = self.sigmoid(z1)
        # OUTPUT OF THE LAST LAYER
        z2 = self.hidout.dot(self.W2) - self.B2
        self.out = self.sigmoid(z2)

    def backward_pass(self, Input, desired):
        """
        Network backward Pass
        """
        # CALCULATE ERROR IN OUTPUT
        out_delta = (desired - self.out) * (self.out * (1 - self.out))
        hid_delta = out_delta.dot(self.W2.T) * (self.hidout * (1 - self.hidout))
        
        # UPDATE WEIGHTS BY GRADIENTS
        self.W2 += (self.hidout.T.dot(out_delta) * self.lrate)
        self.B2 += (-1 * self.lrate * out_delta)
        self.W1 += (Input.T.dot(hid_delta) * self.lrate)
        self.B1 += (-1 * self.lrate * hid_delta)

    def decode(self, w):
        """
        Reshape Weights and Biases to 2D Arrays
        """
        # SIZE OF INPUT-HIDDEN AND HIDDEN-OUTPUT WEIGHTS
        w_layer1_size = self.topology[0] * self.topology[1]
        w_layer2_size = self.topology[1] * self.topology[2]
        
        # INPUT-HIDDEN WEIGHTS
        w_layer1 = w[0:w_layer1_size]
        self.W1 = np.reshape(w_layer1, (self.topology[0], self.topology[1]))
        
        # HIDDEN-OUTPUT WEIGHTS
        w_layer2 = w[w_layer1_size: w_layer1_size + w_layer2_size]
        self.W2 = np.reshape(w_layer2, (self.topology[1], self.topology[2]))
        
        # BIASES 
        self.B1 = w[w_layer1_size + w_layer2_size :w_layer1_size + w_layer2_size + self.topology[1]]
        self.B2 = w[w_layer1_size + w_layer2_size + self.topology[1] :w_layer1_size + w_layer2_size + self.topology[1] + self.topology[2]]

    def encode(self):
        """
        Reshape weights to 1D Array
        """
        
        # RESHAPE TO 1D ARRAY
        w1 = self.W1.ravel()
        w2 = self.W2.ravel()

        # CONCATENATE WEIGHTS
        w = np.concatenate([w1, w2, self.B1, self.B2])
        return w

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
        self.decode(w)

        # INIT VARIABLES
        size = data.shape[0]
        Input = np.zeros((1, self.topology[0]))
        fx = np.zeros((size,self.topology[2]))
        
        # READ DATA ROW BY ROW AND CARRY OUT FORWARD PASS
        for i in range(0, size):
            Input = data[i, 0:self.topology[0]]
            self.forward_pass(Input)
            fx[i] = self.out

        return fx

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
        data = self.train_data
        y = data[:, self.topology[0]: self.topology[0] + self.topology[2]]
        fx = self.generate_output(data, w)
        return self.calculate_rmse(fx, y)