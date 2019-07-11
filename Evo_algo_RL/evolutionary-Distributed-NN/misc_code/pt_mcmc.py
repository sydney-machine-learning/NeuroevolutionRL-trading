# !/usr/bin/python
from __future__ import division
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import random
import time
import math
import random
import os


# --------------------------------------------- Basic Neural Network Class ---------------------------------------------

class Network(object):

    def __init__(self, topology, train_data, test_data, learn_rate = 0.5, alpha = 0.1):
        self.topology = topology  # NN topology [input, hidden, output]
        np.random.seed(int(time.time()))
        self.train_data = train_data
        self.test_data = test_data
        self.W1 = np.random.randn(self.topology[0], self.topology[1]) / np.sqrt(self.topology[0])
        self.B1 = np.random.randn(1, self.topology[1]) / np.sqrt(self.topology[1])  # bias first layer
        self.W2 = np.random.randn(self.topology[1], self.topology[2]) / np.sqrt(self.topology[1])
        self.B2 = np.random.randn(1, self.topology[2]) / np.sqrt(self.topology[1])  # bias second layer
        self.hidout = np.zeros((1, self.topology[1]))  # output of first hidden layer
        self.out = np.zeros((1, self.topology[2]))  # output last layer

    @staticmethod
    def sigmoid(x):
        # x = x.astype(np.float128)
        return 1 / (1 + np.exp(-x))

    def sample_er(self, actualout):
        error = np.subtract(self.out, actualout)
        sqerror = np.sum(np.square(error)) / self.topology[2]
        return sqerror

    def calculate_rmse(self, actual, targets):
        return np.sqrt((np.square(np.subtract(np.absolute(actual), np.absolute(targets)))).mean())

    def sample_ad(self, actualout):
        error = np.subtract(self.out, actualout)
        mod_error = np.sum(np.abs(error)) / self.topology[2]
        return mod_error

    def forward_pass(self, X):
        z1 = X.dot(self.W1) - self.B1
        self.hidout = self.sigmoid(z1)  # output of first hidden layer
        z2 = self.hidout.dot(self.W2) - self.B2
        self.out = self.sigmoid(z2)  # output second hidden layer

    def backward_pass(self, Input, desired):
        out_delta = (desired - self.out) * (self.out * (1 - self.out))
        hid_delta = out_delta.dot(self.W2.T) * (self.hidout * (1 - self.hidout))
        self.W2 += (self.hidout.T.dot(out_delta) * self.lrate)
        self.B2 += (-1 * self.lrate * out_delta)
        self.W1 += (Input.T.dot(hid_delta) * self.lrate)
        self.B1 += (-1 * self.lrate * hid_delta)

    def decode(self, w):
        w_layer1_size = self.topology[0] * self.topology[1]
        w_layer2_size = self.topology[1] * self.topology[2]
        w_layer1 = w[0:w_layer1_size]
        self.W1 = np.reshape(w_layer1, (self.topology[0], self.topology[1]))
        w_layer2 = w[w_layer1_size: w_layer1_size + w_layer2_size]
        self.W2 = np.reshape(w_layer2, (self.topology[1], self.topology[2]))
        self.B1 = w[w_layer1_size + w_layer2_size :w_layer1_size + w_layer2_size + self.topology[1]]
        self.B2 = w[w_layer1_size + w_layer2_size + self.topology[1] :w_layer1_size + w_layer2_size + self.topology[1] + self.topology[2]]

    def encode(self):
        w1 = self.W1.ravel()
        w2 = self.W2.ravel()
        w = np.concatenate([w1, w2, self.B1, self.B2])
        return w

    @staticmethod
    def scale_data(data, maxout=1, minout=0, maxin=1, minin=0):
        attribute = data[:]
        attribute = minout + (attribute - minin)*((maxout - minout)/(maxin - minin))
        return attribute

    @staticmethod
    def denormalize(data, indices, maxval, minval):
        for i in range(len(indices)):
            index = indices[i]
            attribute = data[:, index]
            attribute = Network.scale_data(attribute, maxout=maxval[i], minout=minval[i], maxin=1, minin=0)
            data[:, index] = attribute
        return data

    @staticmethod
    def softmax(fx):
        ex = np.exp(fx)
        sum_ex = np.sum(ex, axis = 1)
        sum_ex = np.multiply(np.ones(ex.shape), sum_ex[:, np.newaxis])
        probability = np.divide(ex, sum_ex)
        return probability

    @staticmethod
    def calculate_accuracy(actual, targets):
        out = np.argmax(targets, axis=1)
        y_out = np.argmax(actual, axis=1)
        count = 0
        for index in range(y_out.shape[0]):
            if out[index] == y_out[index]:
                count += 1
        accuracy = float(count)/y_out.shape[0] * 100
        return accuracy

    @staticmethod
    def calculate_rmse(actual, targets):
        return np.sqrt((np.square(np.subtract(np.absolute(actual), np.absolute(targets)))).mean())

    def generate_output(self, data, w):  # BP with SGD (Stocastic BP)
        self.decode(w)  # method to decode w into W1, W2, B1, B2.
        size = data.shape[0]
        Input = np.zeros((1, self.topology[0]))  # temp hold input
        fx = np.zeros((size,self.topology[2]))
        for i in range(0, size):  # to see what fx is produced by your current weight update
            Input = data[i, 0:self.topology[0]]
            self.forward_pass(Input)
            fx[i] = self.out
        return fx

    def evaluate_fitness(self, w):  # BP with SGD (Stocastic BP
        data = self.train_data
        y = data[:, self.topology[0]: self.topology[0] + self.topology[2]]
        fx = self.generate_output(data, w)
        return self.calculate_rmse(fx, y)


class Replica(multiprocessing.Process):
    def __init__(self, num_samples, burn_in, topology, train_data, test_data, directory, temperature, swap_interval, parameter_queue, problem_type, main_process, event):
        # MULTIPROCESSING CLASS CONSTRUCTOR
        multiprocessing.Process.__init__(self)
        self.process_id = temperature
        self.parameter_queue = parameter_queue
        self.signal_main = main_process
        self.event =  event
        #PARALLEL TEMPERING VARIABLES
        self.temperature = temperature
        self.swap_interval = swap_interval
        self.burn_in = burn_in
        # MCMC VARIABLES
        self.num_samples = num_samples
        self.topology = topology
        self.train_data = train_data
        self.test_data = test_data
        self.problem_type = problem_type
        self.directory = directory
        self.w_size = (topology[0] * topology[1]) + (topology[1] * topology[2]) + topology[1] + topology[2]
        self.neural_network = Network(topology, train_data, test_data)
        self.initialize_sampling_parameters()
        self.create_directory(directory)

    def fitness_function(self, x):
        fitness = self.neural_network.evaluate_fitness(x)
        return fitness

    def initialize_sampling_parameters(self):
        self.w_stepsize = 0.05
        self.eta_stepsize = 0.01
        self.sigma_squared = 36
        self.nu_1 = 0
        self.nu_2 = 0
        self.start_time = time.time()

    @staticmethod
    def convert_time(secs):
        if secs >= 60:
            mins = str(int(secs/60))
            secs = str(int(secs%60))
        else:
            secs = str(int(secs))
            mins = str(00)
        if len(mins) == 1:
            mins = '0'+mins
        if len(secs) == 1:
            secs = '0'+secs
        return [mins, secs]

    @staticmethod
    def create_directory(directory):
        if not os.path.isdir(directory):
            os.mkdir(directory)

    @staticmethod
    def multinomial_likelihood(neural_network, data, weights, temperature):
        y = data[:, neural_network.topology[0]: neural_network.topology[0] + neural_network.topology[2]]
        fx = neural_network.generate_output(data, weights)
        rmse = Network.calculate_rmse(fx, y) # Can be replaced by calculate_nmse function for reporting NMSE
        probability = neural_network.softmax(fx)
        loss = 0
        for index_1 in range(y.shape[0]):
            for index_2 in range(y.shape[1]):
                if y[index_1, index_2] == 1:
                    loss += np.log(probability[index_1, index_2])
        accuracy = Network.calculate_accuracy(fx, y)
        return [loss/temperature, rmse, accuracy]

    def classification_prior(self, sigma_squared, weights):
        h = self.topology[1]  # number hidden neurons
        d = self.topology[0]
        part_1 = -1 * ((d * h + h + 2) / 2) * np.log(sigma_squared)
        part_2 = 1 / (2 * sigma_squared) * (sum(np.square(weights)))
        log_loss = part_1 - part_2
        return log_loss

    @staticmethod
    def gaussian_likelihood(neural_network, data, weights, tausq, temperature):
        desired = data[:, neural_network.topology[0]: neural_network.topology[0] + neural_network.topology[2]]
        prediction = neural_network.generate_output(data, weights)
        rmse = Network.calculate_rmse(prediction, desired)
        loss = -0.5 * np.log(2 * np.pi * tausq) - 0.5 * np.square(desired - prediction) / tausq
        return [np.sum(loss)/temperature, rmse]


    def gaussian_prior(self, sigma_squared, nu_1, nu_2, weights, tausq):
        h = self.topology[1]  # number hidden neurons
        d = self.topology[0]
        part1 = -1 * ((d * h + h + 2) / 2) * np.log(sigma_squared)
        part2 = 1 / (2 * sigma_squared) * (sum(np.square(weights)))
        log_loss = part1 - part2 - (1 + nu_1) * np.log(tausq) - (nu_2 / tausq)
        return log_loss

    def likelihood_function(self, neural_network, data, weights, tau, temperature):
        if self.problem_type == 'regression':
            likelihood, rmse = self.gaussian_likelihood(neural_network, data, weights, tau, temperature)
            return likelihood, rmse, None
        elif self.problem_type == 'classification':
            likelihood, rmse, accuracy = self.multinomial_likelihood(neural_network, data, weights, temperature)
            return likelihood, rmse, accuracy

    def prior_function(self, weights, tau):
        if self.problem_type == 'regression':
            loss = self.gaussian_prior(self.sigma_squared, self.nu_1, self.nu_2, weights, tau)
        elif self.problem_type == 'classification':
            loss = self.classification_prior(self.sigma_squared, weights)
        try:
            return loss
        except Exception as e:
            print(self.problem_type)
            raise(e)

    def evaluate_proposal(self, neural_network, train_data, test_data, weights_proposal, tau_proposal, likelihood_current, prior_current):
        accept = False
        likelihood_ignore, rmse_test_proposal, acc_test = self.likelihood_function(neural_network, test_data, weights_proposal, tau_proposal, self.temperature)
        likelihood_proposal, rmse_train_proposal, acc_train = self.likelihood_function(neural_network, train_data, weights_proposal, tau_proposal, self.temperature)
        prior_proposal = self.prior_function(weights_proposal, tau_proposal)
        difference_likelihood = likelihood_proposal - likelihood_current
        difference_prior = prior_proposal - prior_current
        mh_ratio = min(1, np.exp(min(709, difference_likelihood + difference_prior)))
        u = np.random.uniform(0,1)
        if u < mh_ratio:
            accept = True
            likelihood_current = likelihood_proposal
            prior_proposal = prior_current
        if acc_train == None:
            return accept, rmse_train_proposal, rmse_test_proposal, likelihood_current, prior_current
        else:
            return accept, rmse_train_proposal, rmse_test_proposal, acc_train, acc_test, likelihood_current, prior_current

    def run(self):
        save_knowledge = True
        train_rmse_file = open(self.directory+'/train_rmse_'+str(self.temperature)+'.csv', 'w')
        test_rmse_file = open(self.directory+'/test_rmse_'+str(self.temperature)+'.csv', 'w')
        if self.problem_type == 'classification':
            train_acc_file = open(self.directory+'/train_acc_'+str(self.temperature)+'.csv', 'w')
            test_acc_file = open(self.directory+'/test_acc_'+str(self.temperature)+'.csv', 'w')
        weights_initial = np.random.randn(self.w_size)

        # ------------------- initialize MCMC
        self.start_time = time.time()

        train_size = self.train_data.shape[0]
        test_size = self.test_data.shape[0]
        y_test = self.test_data[:, self.topology[0]: self.topology[0] + self.topology[2]]
        y_train = self.train_data[:, self.topology[0]: self.topology[0] + self.topology[2]]
        weights_current = weights_initial.copy()
        weights_proposal = weights_initial.copy()
        prediction_train = self.neural_network.generate_output(self.train_data, weights_current)
        prediction_test = self.neural_network.generate_output(self.test_data, weights_current)
        eta = np.log(np.var(prediction_train - y_train))
        tau_proposal = np.exp(eta)
        prior = self.prior_function(weights_current, tau_proposal)
        [likelihood, rmse_train, acc_train] = self.likelihood_function(self.neural_network, self.train_data, weights_current, tau_proposal, self.temperature)

        rmse_test = Network.calculate_rmse(prediction_test, y_test)
        if problem_type == 'classification':
            acc_test = Network.calculate_accuracy(prediction_test, y_test)

        # save values into previous variables
        rmse_train_current = rmse_train
        rmse_test_current = rmse_test
        num_accept = 0
        if self.problem_type == 'classification':
            acc_test_current = acc_test
            acc_train_current = acc_train

        if save_knowledge:
            np.savetxt(train_rmse_file, [rmse_train_current])
            np.savetxt(test_rmse_file, [rmse_test_current])
            if problem_type == 'classification':
                np.savetxt(train_acc_file, [acc_train_current])
                np.savetxt(test_acc_file, [acc_test_current])

        # start sampling
        for sample in range(1, self.num_samples):
            # Evaluate population proposal
            weights_proposal = weights_current + np.random.normal(0, self.w_stepsize, self.w_size)
            eta_proposal = eta + np.random.normal(0, self.eta_stepsize, 1)
            tau_proposal = np.exp(eta_proposal)
            if self.problem_type == 'classification':
                accept, rmse_train, rmse_test, acc_train, acc_test, likelihood, prior = self.evaluate_proposal(self.neural_network, self.train_data, self.test_data, weights_proposal, tau_proposal, likelihood, prior)
            else:
                accept, rmse_train, rmse_test, likelihood, prior = self.evaluate_proposal(self.neural_network, self.train_data, self.test_data, weights_proposal, tau_proposal, likelihood, prior)

            if accept:
                num_accept += 1
                weights_current = weights_proposal
                eta = eta_proposal
                # save values into previous variables
                rmse_train_current = rmse_train
                rmse_test_current = rmse_test
                if self.problem_type == 'classification':
                    acc_train_current = acc_train
                    acc_test_current = acc_test

            if save_knowledge:
                np.savetxt(train_rmse_file, [rmse_train_current])
                np.savetxt(test_rmse_file, [rmse_test_current])
                if problem_type == 'classification':
                    np.savetxt(train_acc_file, [acc_train_current])
                    np.savetxt(test_acc_file, [acc_test_current])

            #SWAPPING PREP
            # print sample
            if (sample % self.swap_interval == 0 and sample != 0 ):
                print('\nTemperature: {} Swapping weights: {}'.format(self.temperature, weights_current[:2]))
                param = np.concatenate([weights_current, np.asarray([eta]).reshape(1), np.asarray([likelihood*self.temperature]),np.asarray([self.temperature])])
                self.parameter_queue.put(param)
                self.signal_main.set()
                self.event.wait()
                # print(sample, self.temperature)
                # retrieve parameters fom queues if it has been swapped
                if not self.parameter_queue.empty() :
                    try:
                        result =  self.parameter_queue.get()
                        weights_current = result[0:self.w_size]
                        eta = result[self.w_size]
                        likelihood = result[self.w_size+1]/self.temperature
                        print('Temperature: {} Swapped weights: {}'.format(self.temperature, weights_current[:2]))
                    except:
                        print ('error')
                else:
                    print("Khali")
                self.event.clear()
            elapsed_time = ":".join(Replica.convert_time(time.time() - self.start_time))
            fx = self.neural_network.generate_output(self.train_data, weights_current)
            y = self.train_data[:, self.neural_network.topology[0]: self.neural_network.topology[0] + self.neural_network.topology[2]]
            acc_train_current = Network.calculate_accuracy(fx,y)

            if self.problem_type == 'regression':
                print("Temperature: {:.2f} Sample: {:d}, Best Fitness: {:.4f}, Accuracy: {:.4f}, Time Elapsed: {:s}".format(self.temperature, sample, rmse_train_current, acc_train_current, elapsed_time))
            else:
                print("Temperature: {:.2f} Sample: {:d}, Best Fitness: {:.4f}, Proposal: {:.4f}, Time Elapsed: {:s}".format(self.temperature, sample, acc_train_current, acc_train, elapsed_time))

        elapsed_time = time.time() - self.start_time
        accept_ratio = num_accept/num_samples * 100

        # Close the files
        train_rmse_file.close()
        test_rmse_file.close()
        if self.problem_type == 'classification':
            train_acc_file.close()
            test_acc_file.close()
        print("Temperature: {} Done! with accept ratio: {}".format(self.temperature, accept_ratio))


class ParallelTempering(object):

    def __init__(self, burn_in, train_data, test_data, topology, num_chains, max_temp, num_samples, swap_interval, path, problem_type, geometric=True):
        #FNN Chain variables
        self.train_data = train_data
        self.test_data = test_data
        self.topology = topology
        self.num_param = (topology[0] * topology[1]) + (topology[1] * topology[2]) + topology[1] + topology[2]
        self.problem_type = problem_type
        #Parallel Tempering variables
        self.burn_in = burn_in
        self.swap_interval = swap_interval
        self.path = path
        self.max_temp = max_temp
        self.num_swap = 0
        self.total_swap_proposals = 0
        self.num_chains = num_chains
        self.chains = []
        self.temperatures = []
        self.num_samples = int(num_samples/self.num_chains)
        self.geometric = geometric
        # create queues for transfer of parameters between process chain
        self.parameter_queue = [multiprocessing.Queue() for i in range(num_chains)]
        self.chain_queue = multiprocessing.JoinableQueue()
        self.wait_chain = [multiprocessing.Event() for i in range (self.num_chains)]
        self.event = [multiprocessing.Event() for i in range (self.num_chains)]
        make_directory(path)

    def default_beta_ladder(self, ndim, ntemps, Tmax): #https://github.com/konqr/ptemcee/blob/master/ptemcee/sampler.py
        """
        Returns a ladder of :math:`\beta \equiv 1/T` under a geometric spacing that is determined by the
        arguments ``ntemps`` and ``Tmax``.  The temperature selection algorithm works as follows:
        Ideally, ``Tmax`` should be specified such that the tempered posterior looks like the prior at
        this temperature.  If using adaptive parallel tempering, per `arXiv:1501.05823
        <http://arxiv.org/abs/1501.05823>`_, choosing ``Tmax = inf`` is a safe bet, so long as
        ``ntemps`` is also specified.
        :param ndim:
            The number of dimensions in the parameter space.
        :param ntemps: (optional)
            If set, the number of temperatures to generate.
        :param Tmax: (optional)
            If set, the maximum temperature for the ladder.
        Temperatures are chosen according to the following algorithm:
        * If neither ``ntemps`` nor ``Tmax`` is specified, raise an exception (insufficient
          information).
        * If ``ntemps`` is specified but not ``Tmax``, return a ladder spaced so that a Gaussian
          posterior would have a 25% temperature swap acceptance ratio.
        * If ``Tmax`` is specified but not ``ntemps``:
          * If ``Tmax = inf``, raise an exception (insufficient information).
          * Else, space chains geometrically as above (for 25% acceptance) until ``Tmax`` is reached.
        * If ``Tmax`` and ``ntemps`` are specified:
          * If ``Tmax = inf``, place one chain at ``inf`` and ``ntemps-1`` in a 25% geometric spacing.
          * Else, use the unique geometric spacing defined by ``ntemps`` and ``Tmax``.
        """

        if type(ndim) != int or ndim < 1:
            raise ValueError('Invalid number of dimensions specified.')
        if ntemps is None and Tmax is None:
            raise ValueError('Must specify one of ``ntemps`` and ``Tmax``.')
        if Tmax is not None and Tmax <= 1:
            raise ValueError('``Tmax`` must be greater than 1.')
        if ntemps is not None and (type(ntemps) != int or ntemps < 1):
            raise ValueError('Invalid number of temperatures specified.')

        tstep = np.array([25.2741, 7., 4.47502, 3.5236, 3.0232,
                          2.71225, 2.49879, 2.34226, 2.22198, 2.12628,
                          2.04807, 1.98276, 1.92728, 1.87946, 1.83774,
                          1.80096, 1.76826, 1.73895, 1.7125, 1.68849,
                          1.66657, 1.64647, 1.62795, 1.61083, 1.59494,
                          1.58014, 1.56632, 1.55338, 1.54123, 1.5298,
                          1.51901, 1.50881, 1.49916, 1.49, 1.4813,
                          1.47302, 1.46512, 1.45759, 1.45039, 1.4435,
                          1.4369, 1.43056, 1.42448, 1.41864, 1.41302,
                          1.40761, 1.40239, 1.39736, 1.3925, 1.38781,
                          1.38327, 1.37888, 1.37463, 1.37051, 1.36652,
                          1.36265, 1.35889, 1.35524, 1.3517, 1.34825,
                          1.3449, 1.34164, 1.33847, 1.33538, 1.33236,
                          1.32943, 1.32656, 1.32377, 1.32104, 1.31838,
                          1.31578, 1.31325, 1.31076, 1.30834, 1.30596,
                          1.30364, 1.30137, 1.29915, 1.29697, 1.29484,
                          1.29275, 1.29071, 1.2887, 1.28673, 1.2848,
                          1.28291, 1.28106, 1.27923, 1.27745, 1.27569,
                          1.27397, 1.27227, 1.27061, 1.26898, 1.26737,
                          1.26579, 1.26424, 1.26271, 1.26121,
                          1.25973])

        if ndim > tstep.shape[0]:
            # An approximation to the temperature step at large
            # dimension
            tstep = 1.0 + 2.0*np.sqrt(np.log(4.0))/np.sqrt(ndim)
        else:
            tstep = tstep[ndim-1]

        appendInf = False
        if Tmax == np.inf:
            appendInf = True
            Tmax = None
            ntemps = ntemps - 1

        if ntemps is not None:
            if Tmax is None:
                # Determine Tmax from ntemps.
                Tmax = tstep ** (ntemps - 1)
        else:
            if Tmax is None:
                raise ValueError('Must specify at least one of ``ntemps'' and '
                                 'finite ``Tmax``.')

            # Determine ntemps from Tmax.
            ntemps = int(np.log(Tmax) / np.log(tstep) + 2)

        betas = np.logspace(0, -np.log10(Tmax), ntemps)
        if appendInf:
            # Use a geometric spacing, but replace the top-most temperature with
            # infinity.
            betas = np.concatenate((betas, [0]))

        return betas

    def assign_temperatures(self):
        #Geometric Spacing
        if self.geometric is True:
            betas = self.default_beta_ladder(2, ntemps=self.num_chains, Tmax=self.max_temp)
            self.temperatures = [np.inf if beta == 0 else 1.0/beta for beta in betas]
        #Linear Spacing
        else:
            temp = 2
            for i in range(0,self.num_chains):
                self.temperatures.append(temp)
                temp += 2.5 #(self.maxtemp/self.num_chains)
                print (self.temperatures[i])

    def initialize_chains(self):
        self.assign_temperatures()
        weights = np.random.randn(self.num_param)
        for chain in range(0, self.num_chains):
            self.chains.append(Replica(self.num_samples, self.burn_in, self.topology, self.train_data, self.test_data, self.path, self.temperatures[chain], self.swap_interval, self.parameter_queue[chain], self.problem_type, main_process=self.wait_chain[chain], event=self.event[chain]))

    def swap_procedure(self, parameter_queue_1, parameter_queue_2):
        if not parameter_queue_2.empty() and not parameter_queue_1.empty():
            param_1 = parameter_queue_1.get()
            param_2 = parameter_queue_2.get()
            w_1 = param_1[0:self.num_param]
            eta_1 = param_1[self.num_param]
            likelihood_1 = param_1[self.num_param+1]
            T_1 = param_1[self.num_param+2]
            w_2 = param_2[0:self.num_param]
            eta_2 = param_2[self.num_param]
            likelihood_2 = param_2[self.num_param+1]
            T_2 = param_2[self.num_param+2]
            #SWAPPING PROBABILITIES
            try:
                swap_proposal =  min(1,0.5*np.exp(likelihood_2 - likelihood_1))
            except OverflowError:
                swap_proposal = 1
            u = np.random.uniform(0,1)
            if u < swap_proposal:
                swapped = True
                self.num_swap += 1
                param_temp =  param_1
                param_1 = param_2
                param_2 = param_temp
                print("Swapped {}, {}".format(param_1[:2], param_2[:2]))
            else:
                print("No swapping!!")
                swapped = False
            self.total_swap_proposals += 1
            return param_1, param_2, swapped
        else:
            print("No Swapping occured")
            self.total_swap_proposals += 1
            raise Exception('empty queue')
            return

    def plot_figure(self, list, title):

        list_points =  list

        fname = self.path
        width = 9

        font = 9

        fig = plt.figure(figsize=(10, 12))
        ax = fig.add_subplot(111)


        slen = np.arange(0,len(list),1)

        fig = plt.figure(figsize=(10,12))
        ax = fig.add_subplot(111)
        ax.spines['top'].set_color('none')
        ax.spines['bottom'].set_color('none')
        ax.spines['left'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
        ax.set_title(' Posterior distribution', fontsize=  font+2)#, y=1.02)

        ax1 = fig.add_subplot(211)

        n, rainbins, patches = ax1.hist(list_points,  bins = 20,  alpha=0.5, facecolor='sandybrown', density=False)


        color = ['blue','red', 'pink', 'green', 'purple', 'cyan', 'orange','olive', 'brown', 'black']

        ax1.grid(True)
        ax1.set_ylabel('Frequency',size= font+1)
        ax1.set_xlabel('Parameter values', size= font+1)

        ax2 = fig.add_subplot(212)

        list_points = np.asarray(np.split(list_points,  self.num_chains ))




        ax2.set_facecolor('#f2f2f3')
        ax2.plot( list_points.T , label=None)
        ax2.set_title(r'Trace plot',size= font+2)
        ax2.set_xlabel('Samples',size= font+1)
        ax2.set_ylabel('Parameter values', size= font+1)

        fig.tight_layout()
        fig.subplots_adjust(top=0.88)


        plt.savefig(fname + '/' + title  + '_pos_.png', bbox_inches='tight', dpi=300, transparent=False)
        plt.clf()


    def run_chains(self):
        x_test = np.linspace(0,1,num=self.test_data.shape[0])
        x_train = np.linspace(0,1,num=self.train_data.shape[0])
        # only adjacent chains can be swapped therefore, the number of proposals is ONE less num_chains
        swap_proposal = np.ones(self.num_chains-1)
        # create parameter holders for paramaters that will be swapped
        replica_param = np.zeros((self.num_chains, self.num_param))
        likelihood = np.zeros(self.num_chains)
        eta = np.zeros(self.num_chains)
        # Define the starting and ending of MCMC Chains
        start = 0
        end = self.num_samples-1
        print("Num Samples: {}".format(self.num_samples))
        number_exchange = np.zeros(self.num_chains)
        filen = open(self.path + '/num_exchange.txt', 'a')
        #RUN MCMC CHAINS
        for index in range(self.num_chains):
            self.chains[index].start_chain = start
            self.chains[index].end = end

        for index in range(self.num_chains):
            self.chains[index].start()

        swaps_appected_main = 0
        total_swaps_main = 0

        #SWAP PROCEDURE
        while True:
            count = 0
            for index in range(self.num_chains):
                if not self.chains[index].is_alive():
                    count+=1
                    print(str(self.chains[index].temperature) +" Dead")

            if count == self.num_chains:
                break
            print("Waiting")
            timeout_count = 0
            for index in range(0,self.num_chains):
                print("Waiting for chain: {}".format(index+1))
                flag = self.wait_chain[index].wait(timeout=5)
                if flag:
                    print("Signal from chain: {}".format(index+1))
                    timeout_count += 1

            if timeout_count != self.num_chains:
                print("Skipping the swap!")
                continue
            print("Event occured")
            for index in range(0,self.num_chains-1):
                print('starting swap')
                try:
                    param_1, param_2, swapped = self.swap_procedure(self.parameter_queue[index],self.parameter_queue[index+1])
                    self.parameter_queue[index].put(param_1)
                    self.parameter_queue[index+1].put(param_2)
                    if index == 0:
                        if swapped:
                            swaps_appected_main += 1
                        total_swaps_main += 1
                except:
                    print("Nothing Returned by swap method!")
            for index in range (self.num_chains):
                    self.event[index].set()
                    self.wait_chain[index].clear()

        print("Joining processes")

        #JOIN THEM TO MAIN PROCESS
        for index in range(0,self.num_chains):
            self.chains[index].join()
        self.chain_queue.join()

        #GETTING DATA
        burn_in = int(self.num_samples*self.burn_in)
        rmse_train = np.zeros((self.num_chains,self.num_samples - burn_in))
        rmse_test = np.zeros((self.num_chains,self.num_samples - burn_in))
        if self.problem_type == 'classification':
            acc_train = np.zeros((self.num_chains,self.num_samples - burn_in))
            acc_test = np.zeros((self.num_chains,self.num_samples - burn_in))
        accept_ratio = np.zeros((self.num_chains,1))

        for i in range(self.num_chains):

            file_name = self.path+'/test_rmse_'+ str(self.temperatures[i])+ '.csv'
            dat = np.genfromtxt(file_name, delimiter=',')
            rmse_test[i,:] = dat[burn_in:]

            file_name = self.path+'/train_rmse_'+ str(self.temperatures[i])+ '.csv'
            dat = np.genfromtxt(file_name, delimiter=',')
            rmse_train[i,:] = dat[burn_in:]

            if self.problem_type == 'classification':
                file_name = self.path+'/test_acc_'+ str(self.temperatures[i])+ '.csv'
                dat = np.genfromtxt(file_name, delimiter=',')
                acc_test[i,:] = dat[burn_in:]

                file_name = self.path+'/train_acc_'+ str(self.temperatures[i])+ '.csv'
                dat = np.genfromtxt(file_name, delimiter=',')
                acc_train[i,:] = dat[burn_in:]

        rmse_train = rmse_train.reshape(self.num_chains*(self.num_samples - burn_in), 1)
        rmse_test = rmse_test.reshape(self.num_chains*(self.num_samples - burn_in), 1)
        if self.problem_type == 'classification':
            acc_train = acc_train.reshape(self.num_chains*(self.num_samples - burn_in), 1)
            acc_test = acc_test.reshape(self.num_chains*(self.num_samples - burn_in), 1)

        plt.plot(rmse_train[:self.num_samples - burn_in])
        plt.xlabel('samples')
        plt.ylabel('RMSE')
        plt.show()
        plt.clf()

        plt.plot(rmse_train)
        plt.xlabel('samples')
        plt.ylabel('RMSE')
        plt.show()
        plt.clf()

        plt.plot(rmse_test[:self.num_samples - burn_in])
        plt.xlabel('samples')
        plt.ylabel('RMSE')
        plt.show()
        plt.clf()

        plt.plot(rmse_test)
        plt.xlabel('samples')
        plt.ylabel('RMSE')
        plt.show()
        plt.clf()


        if self.problem_type == 'classification':
            plt.plot(acc_train[:self.num_samples - burn_in])
            plt.xlabel('samples')
            plt.ylabel('Accuracy')
            plt.show()
            plt.clf()

            plt.plot(acc_test[:self.num_samples - burn_in])
            plt.xlabel('samples')
            plt.ylabel('Accuracy')
            plt.show()
            plt.clf()

        print("NUMBER OF SWAPS MAIN =", total_swaps_main)
        print("SWAP ACCEPTANCE = ", self.num_swap*100/self.total_swap_proposals," %")
        print("SWAP ACCEPTANCE MAIN = ", swaps_appected_main*100/total_swaps_main," %")

        if self.problem_type == 'classification':
            return (rmse_train, rmse_test, acc_train, acc_test)

        return (rmse_train, rmse_test)

def make_directory(path):
    if not os.path.isdir(path):
        os.makedirs(path)

if __name__ == '__main__':
    # Select problem
    problem = 3

    if problem == 1:
        # Synthetic
        num_samples = 40000
        population_size = 100
        burn_in = 0.2
        num_chains = 10
        max_temp = 20
        swap_interval = 100
        problem_type = 'regression'
        topology = [4, 25, 1]
        problem_name = 'synthetic'
        path = 'results_rw/synthetic_' + str(num_chains) + '_' + str(max_temp)

        train_data_file = '../Datasets/synthetic_data/target_train.csv'
        test_data_file = '../Datasets/synthetic_data/target_test.csv'

        train_data = np.genfromtxt(train_data_file, delimiter=',')
        test_data = np.genfromtxt(test_data_file, delimiter=',')

    elif problem == 2:
        # UJIndoorLoc
        num_samples = 100000
        population_size = 100
        burn_in = 0.2
        num_chains = 10
        max_temp = 20
        swap_interval = 200
        problem_type = 'regression'
        topology = [520, 48, 2]
        problem_name = 'UJIndoorLoc_0'
        path = 'results_rw/UJIndoorLoc_0_' + str(num_chains) + '_' + str(max_temp)

        train_data_file = '../UJIndoorLoc/sourceData/0train.csv'
        test_data_file = '../UJIndoorLoc/sourceData/0test.csv'

        train_data = np.genfromtxt(train_data_file, delimiter=',')[:, :-2]
        test_data = np.genfromtxt(test_data_file, delimiter=',')[:, :-2]

    elif problem == 3:
        #Iris
        num_samples = 80000
        population_size = 100
        burn_in = 0.2
        num_chains = 10
        max_temp = 20
        swap_interval = 80
        problem_type = 'classification'
        topology = [4, 15, 3]
        problem_name = 'Iris'
        path = 'results_rw/Iris' + str(num_chains) + '_' + str(max_temp)

        train_data_file = '../Datasets/Iris/iris-train.csv'
        test_data_file = '../Datasets/Iris/iris-test.csv'

        train_data = np.genfromtxt(train_data_file, delimiter=',')
        test_data = np.genfromtxt(test_data_file, delimiter=',')

    elif problem == 4:
        #Ions
        num_samples = 80000
        # population_size = 200
        burn_in = 0.2
        num_chains = 12
        max_temp = 25
        swap_interval = 80
        problem_type = 'classification'
        topology = [34, 50, 2]
        problem_name = 'Ionosphere'
        path = 'results_rw/Ions' + str(num_chains) + '_' + str(max_temp)

        train_data_file = '../Datasets/Ions/train.csv'
        test_data_file = '../Datasets/Ions/test.csv'

        train_data = np.genfromtxt(train_data_file, delimiter=',')
        test_data = np.genfromtxt(test_data_file, delimiter=',')

    elif problem == 5:
        #Cancer
        num_samples = 2000
        population_size = 50
        burn_in = 0.2
        num_chains = 1
        max_temp = 25
        swap_interval = 100
        problem_type = 'classification'
        topology = [9, 12, 2]
        problem_name = 'Cancer'
        path = 'results/Cancer' + str(num_chains) + '_' + str(max_temp)

        train_data_file = '../Datasets/Cancer/ftrain.txt'
        test_data_file = '../Datasets/Cancer/ftest.txt'

        train_data = np.genfromtxt(train_data_file)
        test_data = np.genfromtxt(test_data_file)

    model = ParallelTempering(burn_in, train_data, test_data, topology, num_chains, max_temp, num_samples, swap_interval, path, problem_type=problem_type)
    model.initialize_chains()
    if problem_type == 'classification':
        rmse_train, rmse_test, acc_train, acc_test = model.run_chains()
        print("Combined Result: ")
        print("Train RMSE: ", rmse_train.mean(), "std: ", rmse_train.std())
        print("Test RMSE: ", rmse_test.mean(), "std: ", rmse_test.std())
        print("Train Accuracy: ", acc_train.mean(), "std: ", acc_train.std())
        print("Test Accuracy: ", acc_test.mean(), "std: ", acc_test.std())
        num_samples = int(num_samples/num_chains)
        burn_in = int(burn_in*num_samples)
        rmse_train = rmse_train[: num_samples - burn_in]
        rmse_test = rmse_test[: num_samples - burn_in]
        acc_train = acc_train[: num_samples - burn_in]
        acc_test = acc_test[: num_samples - burn_in]
        print("\nMain Chain Result: ")
        print("Train RMSE: ", rmse_train.mean(), "std: ", rmse_train.std())
        print("Test RMSE: ", rmse_test.mean(), "std: ", rmse_test.std())
        print("Train Accuracy: ", acc_train.mean(), "std: ", acc_train.std())
        print("Test Accuracy: ", acc_test.mean(), "std: ", acc_test.std())

    else:
        rmse_train, rmse_test = model.run_chains()
        print("Combined Result: ")
        print("Train RMSE: ", rmse_train.mean(), "std: ", rmse_train.std())
        print("Test RMSE: ", rmse_test.mean(), "std: ", rmse_test.std())
        num_samples = int(num_samples/num_chains)
        burn_in = int(burn_in*num_samples)
        rmse_train = rmse_train[: num_samples - burn_in]
        rmse_test = rmse_test[: num_samples - burn_in]
        print("\nMain Chain Result: ")
        print("Train RMSE: ", rmse_train.mean(), "std: ", rmse_train.std())
        print("Test RMSE: ", rmse_test.mean(), "std: ", rmse_test.std())
