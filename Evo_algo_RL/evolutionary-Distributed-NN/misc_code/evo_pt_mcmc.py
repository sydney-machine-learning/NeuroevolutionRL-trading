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
import sys


# --------------------------------------------- Basic Neural Network Class ---------------------------------------------

class Network(object):

    def __init__(self, topology, train_data, test_data, learn_rate = 0.5, alpha = 0.1):
        self.topology = topology  # NN topology [input, hidden, output]
        np.random.seed(int(time.time()))
        self.train_data = train_data
        self.test_data = test_data
        self.lrate = learn_rate

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

class G3PCX(object):
    def __init__(self, population_size, num_variables, max_limits, min_limits, max_evals=500000):
        self.initialize_parameters()
        self.sp_size = self.children + self.family
        self.population = np.random.uniform(min_limits[0], max_limits[0], size=(population_size, num_variables))  #[SpeciesPopulation(num_variables) for count in xrange(population_size)]
        self.sub_pop = np.random.uniform(min_limits[0], max_limits[0], size=(self.sp_size, num_variables))  #[SpeciesPopulation(num_variables) for count in xrange(NPSize)]
        self.fitness = np.zeros(population_size)
        self.sp_fit  = np.zeros(self.sp_size)
        self.best_index = 0
        self.best_fit = 0
        self.worst_index = 0
        self.worst_fit = 0
        self.rand_parents =  self.num_parents
        self.temp_index =  np.arange(0, population_size)
        self.rank =  np.arange(0, population_size)
        self.list = np.arange(0, self.sp_size)
        self.parents = np.arange(0, population_size)
        self.population_size = population_size
        self.num_variables = num_variables
        self.num_evals = 0
        self.max_evals = max_evals
        self.problem = 1

    def fitness_function(self, x):    #  function  (can be any other function, model or even a neural network)
        fit = 0.0
        if self.problem == 1: # rosenbrock
            for j in range(x.size -1):
                fit += (100.0*(x[j]*x[j] - x[j+1])*(x[j]*x[j] - x[j+1]) + (x[j]-1.0)*(x[j]-1.0))
        elif self.problem ==2:  # ellipsoidal - sphere function
            for j in range(x.size):
                fit = fit + ((j+1)*(x[j]*x[j]))
        return fit # note we will maximize fitness, hence minimize error

    def initialize_parameters(self):
        self.epsilon = 1e-40  # convergence
        self.sigma_eta = 0.1
        self.sigma_zeta = 0.1
        self.children = 2
        self.num_parents = 3
        self.family = 2

    def rand_normal(self, mean, stddev):
        if (not G3PCX.n2_cached):
            #choose a point x,y in the unit circle uniformly at random
            x = np.random.uniform(-1,1,1)
            y = np.random.uniform(-1,1,1)
            r = x*x + y*y
            while (r == 0 or r > 1):
                x = np.random.uniform(-1,1,1)
                y = np.random.uniform(-1,1,1)
                r = x*x + y*y
            # Apply Box-Muller transform on x, y
            d = np.sqrt(-2.0*np.log(r)/r)
            n1 = x*d
            G3PCX.n2 = y*d
            # scale and translate to get desired mean and standard deviation
            result = n1*stddev + mean
            G3PCX.n2_cached = True
            return result
        else:
            G3PCX.n2_cached = False
            return G3PCX.n2*stddev + mean

    def evaluate(self):
        self.fitness[0] = self.fitness_function(self.population[0,:])
        self.best_fit = self.fitness[0]
        for i in range(self.population_size):
            self.fitness[i] = self.fitness_function(self.population[i,:])
            if (self.best_fit> self.fitness[i]):
                self.best_fit =  self.fitness[i]
                self.best_index = i
        self.num_evals += 1

    # calculates the magnitude of a vector
    def mod(self, List):
        sum = 0
        for i in range(self.num_variables):
            sum += (List[i] * List[i] )
        return np.sqrt(sum)

    def parent_centric_xover(self, current):
        centroid = np.zeros(self.num_variables)
        tempar1 = np.zeros(self.num_variables)
        tempar2 = np.zeros(self.num_variables)
        temp_rand = np.zeros(self.num_variables)
        d = np.zeros(self.num_variables)
        D = np.zeros(self.num_parents)
        temp1, temp2, temp3 = (0,0,0)
        diff = np.zeros((self.num_parents, self.num_variables))
        for i in range(self.num_variables):
            for u in range(self.num_parents):
                centroid[i]  = centroid[i] +  self.population[self.temp_index[u],i]
        centroid   = centroid / self.num_parents
        # calculate the distace (d) from centroid to the index parent self.temp_index[0]
        # also distance (diff) between index and other parents are computed
        for j in range(1, self.num_parents):
            for i in range(self.num_variables):
                if j == 1:
                    d[i]= centroid[i]  - self.population[self.temp_index[0],i]
                diff[j, i] = self.population[self.temp_index[j], i] - self.population[self.temp_index[0],i]
            if (self.mod(diff[j,:]) < self.epsilon):
                print('Points are very close to each other. Quitting this run')
                return 0
        dist = self.mod(d)
        if (dist < self.epsilon):
            print ("\nError -  points are very close to each other. Quitting this run\n")
            return 0
        # orthogonal directions are computed
        for j in range(1, self.num_parents):
            temp1 = self.inner(diff[j,:] , d )
            if ((self.mod(diff[j,:]) * dist) == 0):
                print("Division by zero")
                temp2 = temp1 / (1)
            else:
                temp2 = temp1 / (self.mod(diff[j,:]) * dist)
            temp3 = 1.0 - np.power(temp2, 2)
            D[j] = self.mod(diff[j]) * np.sqrt(np.abs(temp3))
        D_not = 0.0
        for i in range(1, self.num_parents):
            D_not += D[i]
        D_not /= (self.num_parents - 1) # this is the average of the perpendicular distances from all other parents (minus the index parent) to the index vector
        G3PCX.n2 = 0.0
        G3PCX.n2_cached = False
        for i in range(self.num_variables):
            tempar1[i] = self.rand_normal(0,  self.sigma_eta * D_not) #rand_normal(0, D_not * sigma_eta);
            tempar2[i] = tempar1[i]
        if(np.power(dist, 2) == 0):
            print(" division by zero: part 2")
            tempar2  = tempar1
        else:
            tempar2  = tempar1  - (np.multiply(self.inner(tempar1, d) , d )  ) / np.power(dist, 2.0)
        tempar1 = tempar2
        self.sub_pop[current,:] = self.population[self.temp_index[0],:] + tempar1
        rand_var = self.rand_normal(0, self.sigma_zeta)
        for j in range(self.num_variables):
            temp_rand[j] =  rand_var
        self.sub_pop[current,:] += np.multiply(temp_rand ,  d )
        self.sp_fit[current] = self.fitness_function(self.sub_pop[current,:])
        self.num_evals += 1
        return 1

    def inner(self, ind1, ind2):
        sum = 0.0
        for i in range(self.num_variables):
            sum += (ind1[i] * ind2[i])
        return  sum

    def sort_population(self):
        dbest = 99
        for i in range(self.children + self.family):
            self.list[i] = i
        for i in range(self.children + self.family - 1):
            dbest = self.sp_fit[self.list[i]]
            for j in range(i + 1, self.children + self.family):
                if(self.sp_fit[self.list[j]]  < dbest):
                    dbest = self.sp_fit[self.list[j]]
                    temp = self.list[j]
                    self.list[j] = self.list[i]
                    self.list[i] = temp

    def replace_parents(self): #here the best (1 or 2) individuals replace the family of parents
        for j in range(self.family):
            self.population[ self.parents[j],:]  =  self.sub_pop[ self.list[j],:] # Update population with new species
            fx = self.fitness_function(self.population[ self.parents[j],:])
            self.fitness[self.parents[j]]   =  fx
            self.num_evals += 1

    def family_members(self): #//here a random family (1 or 2) of parents is created who would be replaced by good individuals
        swp = 0
        for i in range(self.population_size):
            self.parents[i] = i
        for i in range(self.family):
            randomIndex = random.randint(0, self.population_size - 1) + i # Get random index in population
            if randomIndex > (self.population_size-1):
                randomIndex = self.population_size-1
            swp = self.parents[randomIndex]
            self.parents[randomIndex] = self.parents[i]
            self.parents[i] = swp

    def find_parents(self): #here the parents to be replaced are added to the temporary subpopulation to assess their goodness against the new solutions formed which will be the basis of whether they should be kept or not
        self.family_members()
        for j in range(self.family):
            self.sub_pop[self.children + j, :] = self.population[self.parents[j],:]
            fx = self.fitness_function(self.sub_pop[self.children + j, :])
            self.sp_fit[self.children + j]  = fx
            self.num_evals += 1

    def random_parents(self ):
        for i in range(self.population_size):
            self.temp_index[i] = i
        swp=self.temp_index[0]
        self.temp_index[0]=self.temp_index[self.best_index]
        self.temp_index[self.best_index]  = swp
        # best is always included as a parent and is the index parent
        # this can be changed for solving a generic problem
        for i in range(1, self.rand_parents):
            index= np.random.randint(self.population_size)+i
            if index > (self.population_size-1):
                index = self.population_size-1
            swp=self.temp_index[index]
            self.temp_index[index]=self.temp_index[i]
            self.temp_index[i]=swp

    def evolve(self, outfile):
        tempfit = 0
        prevfitness = 99
        self.evaluate()
        tempfit= self.fitness[self.best_index]
        while(self.num_evals < self.max_evals):
            tempfit = self.best_fit
            self.random_parents()
            for i in range(self.children):
                tag = self.parent_centric_xover(i)
                if (tag == 0):
                    break
            if tag == 0:
                break
            self.find_parents()
            self.sort_population()
            self.replace_parents()
            self.best_index = 0
            tempfit = self.fitness[0]
            for x in range(1, self.population_size):
                if(self.fitness[x] < tempfit):
                    self.best_index = x
                    tempfit  =  self.fitness[x]
            if self.num_evals % 197 == 0:
                print(self.fitness[self.best_index])
                print(self.num_evals, 'num of evals\n\n\n')
            np.savetxt(outfile, [ self.num_evals, self.best_index, self.best_fit], fmt='%1.5f', newline="\n")
        print(self.sub_pop, '  sub_pop')
        print(self.population[self.best_index], ' best sol')
        print(self.fitness[self.best_index], ' fitness')

class Replica(G3PCX, multiprocessing.Process):
    def __init__(self, num_samples, burn_in, population_size, topology, train_data, test_data, directory, temperature, swap_interval, parameter_queue, problem_type,  main_process, event, max_limit=(-5), min_limit=5):
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
        self.min_limits = np.repeat(min_limit, self.w_size)
        self.max_limits = np.repeat(max_limit, self.w_size)
        self.initialize_sampling_parameters()
        self.create_directory(directory)
        G3PCX.__init__(self, population_size, self.w_size, self.max_limits, self.min_limits)

    def fitness_function(self, x):
        fitness = self.neural_network.evaluate_fitness(x)
        return fitness

    def initialize_sampling_parameters(self):
        self.eta_stepsize = 0.02
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

    @staticmethod
    def classification_prior(sigma_squared, weights):
        part_1 = -1 * ((weights.shape[0]) / 2) * np.log(sigma_squared)
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

    @staticmethod
    def gaussian_prior(sigma_squared, nu_1, nu_2, weights, tausq):
        part1 = -1 * (weights.shape[0] / 2) * np.log(sigma_squared)
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
        # quit()
        return loss

    def evaluate_proposal(self, neural_network, train_data, test_data, weights_proposal, tau_proposal, likelihood_current, prior_current):
        accept = False
        likelihood_ignore, rmse_test_proposal, acc_test = self.likelihood_function(neural_network, test_data, weights_proposal, tau_proposal, self.temperature)
        likelihood_proposal, rmse_train_proposal, acc_train = self.likelihood_function(neural_network, train_data, weights_proposal, tau_proposal, self.temperature)
        prior_proposal = self.prior_function(weights_proposal, tau_proposal)
        difference_likelihood = likelihood_proposal - likelihood_current
        difference_prior = prior_proposal - prior_current
        mh_ratio = min(1, np.exp(min(709, difference_likelihood)))
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
        weights_initial = np.random.uniform(-5, 5, self.w_size)

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

        tempfit = 0
        self.evaluate()
        tempfit = self.fitness[self.best_index]
        writ = 0
        if save_knowledge:
            train_rmse_file.write(str(rmse_train_current)+"\n")
            test_rmse_file.write(str(rmse_test_current)+"\n")
            if problem_type == 'classification':
                train_acc_file.write(str(acc_train_current)+"\n")
                test_acc_file.write(str(acc_test_current)+"\n")
                writ += 1

        # start sampling
        for sample in range(1, self.num_samples):
            tempfit = self.best_fit
            self.random_parents()
            for i in range(self.children):
                tag = self.parent_centric_xover(i)
                if (tag == 0):
                    break
            if tag == 0:
                break
            self.find_parents()
            self.sort_population()
            self.replace_parents()
            self.best_index = 0
            tempfit = self.fitness[0]
            for x in range(1, self.population_size):
                if(self.fitness[x] < tempfit):
                    self.best_index = x
                    tempfit  =  self.fitness[x]
            # Evaluate population proposal
            # for index in range(self.population_size):
            weights_proposal = self.population[self.best_index]
            eta_proposal = eta + np.random.normal(0, self.eta_stepsize, 1)
            tau_proposal = np.exp(eta_proposal)
            if self.problem_type == 'classification':
                accept, rmse_train, rmse_test, acc_train, acc_test, likelihood, prior = self.evaluate_proposal(self.neural_network, self.train_data, self.test_data, weights_proposal, tau_proposal, likelihood, prior)
                #print rmse_train, rmse_test, acc_train, acc_test, sample 

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

                print(acc_train, sample)

            if save_knowledge:
                train_rmse_file.write(str(rmse_train_current)+"\n")
                test_rmse_file.write(str(rmse_test_current)+"\n")
                if problem_type == 'classification':
                    train_acc_file.write(str(acc_train_current)+"\n")
                    test_acc_file.write(str(acc_test_current)+"\n")
                    writ += 1
                    # print(writ, acc_test_current, "writing acc")

            #SWAPPING PREP
            # print sample
            if (sample % self.swap_interval == 0 and sample != 0 ):
                # print('\nTemperature: {} Swapping weights: {}'.format(self.temperature, weights_current[:2]))
                param = np.concatenate([weights_current, np.asarray([eta]).reshape(1), np.asarray([likelihood*self.temperature]),np.asarray([self.temperature])])
                self.parameter_queue.put(param)
                self.signal_main.set()
                self.event.wait()
                # print(sample, self.temperature)
                # retrieve parameters fom queues if it has been swapped
                if not self.parameter_queue.empty() :
                    try:
                        result =  self.parameter_queue.get()
                        #print(self.temperature, w, 'param after swap')
                        weights_current = result[0:self.w_size]
                        self.population[self.best_index] = weights_current
                        self.fitness[self.best_index] = self.fitness_function(weights_current)
                        eta = result[self.w_size]
                        likelihood = result[self.w_size+1]/self.temperature
                        # likelihood = self.likelihood_function(self.neural_network, self.train_data, weights_current, np.exp(eta))
                        print('Temperature: {} Swapped weights: {}'.format(self.temperature, weights_current[:2]))
                    except:
                        print ('error')
                else:
                    print("Khali")
                self.event.clear()
            elapsed_time = ":".join(Replica.convert_time(time.time() - self.start_time))

            print("Temperature: {:.2f} Sample: {:d}, Best Fitness: {:.4f}, Proposal: {:.4f}, Time Elapsed: {:s}".format(self.temperature, sample, rmse_train_current, rmse_train, elapsed_time))

        elapsed_time = time.time() - self.start_time
        accept_ratio = num_accept/num_samples
        print("Written {} values for Accuracies".format(writ))
        # Close the files
        train_rmse_file.close()
        test_rmse_file.close()
        print("Temperature: {} Done!".format(self.temperature))

class EvolutionaryParallelTempering(object):

    def __init__(self, burn_in, train_data, test_data, topology, num_chains, max_temp, num_samples, swap_interval, path, population_size, problem_type='regression', geometric=True):
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
        self.population_size = population_size
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
            self.chains.append(Replica(self.num_samples, self.burn_in, self.population_size, self.topology, self.train_data, self.test_data, self.path, self.temperatures[chain], self.swap_interval, self.parameter_queue[chain], self.problem_type, main_process=self.wait_chain[chain], event=self.event[chain]))

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
        #plt.show()
        plt.savefig(self.path+'/1.png') 
        plt.clf()

        plt.plot(rmse_train)
        plt.xlabel('samples')
        plt.ylabel('RMSE')
        plt.savefig(self.path+'/2.png') 
        #plt.show()
        plt.clf()

        plt.plot(rmse_test[:self.num_samples - burn_in])
        plt.xlabel('samples')
        plt.ylabel('RMSE')
        plt.savefig(self.path+'/3.png') 
        #plt.show()
        plt.clf()

        plt.plot(rmse_test)
        plt.xlabel('samples')
        plt.ylabel('RMSE')
        #plt.show()
        plt.savefig(self.path+'/3.png') 
        plt.clf()


        if self.problem_type == 'classification':
            plt.plot(acc_train[:self.num_samples - burn_in])
            plt.xlabel('samples')
            plt.ylabel('Accuracy')
            plt.savefig(self.path+'/4.png')
            #plt.show()
            plt.clf()

            plt.plot(acc_test[:self.num_samples - burn_in])
            plt.xlabel('samples')
            plt.ylabel('RMSE')
            plt.savefig(self.path+'/5.png')
            #plt.show()
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





    if(len(sys.argv)!=5):
        sys.exit('not right input format.  ')



    problem = int(sys.argv[1])  # get input

    num_chains = int(sys.argv[2])

    population_size = int(sys.argv[3])

    swap_interval = int(sys.argv[4])

    print(problem, num_chains, population_size, swap_interval)


 


    burn_in = 0.4  
    max_temp = 5 

    separate_flag = False # for further data processing in some problems 


    if problem == 1:
        # Synthetic
        num_samples = 5000 
        problem_type = 'regression'
        topology = [4, 25, 1]
        problem_name = 'synthetic'
        #path = 'results/synthetic_' + str(num_chains) + '_' + str(max_temp)

        train_data_file = '../Datasets/synthetic_data/target_train.csv'
        test_data_file = '../Datasets/synthetic_data/target_test.csv'

        train_data = np.genfromtxt(train_data_file, delimiter=',')
        test_data = np.genfromtxt(test_data_file, delimiter=',')
 
    elif problem == 2:
        #Iris
        num_samples = 5000 
        problem_type = 'classification'
        topology = [4, 10, 3]
        problem_name = 'Iris'
        #path = 'results/Iris' + str(num_chains) + '_' + str(max_temp)

        #train_data_file = '../Datasets/Iris/iris-train.csv'
        #test_data_file = '../Datasets/Iris/iris-test.csv'

        train_data_file = '../Datasets/data_classification/Iris/iris-train.csv'
        test_data_file = '../Datasets/data_classification/Iris/iris-test.csv'

        train_data = np.genfromtxt(train_data_file, delimiter=',')
        test_data = np.genfromtxt(test_data_file, delimiter=',')

    elif problem == 3:
        #Ions
        num_samples = 4000  
        problem_type = 'classification'
        topology = [34, 50, 2]
        problem_name = 'Ionosphere'
        #path = 'results/Ions' + str(num_chains) + '_' + str(max_temp)

        train_data_file = '../Datasets/data_classification/Ions/ftrain.csv'
        test_data_file = '../Datasets/data_classification/Ions/ftest.csv'

        train_data = np.genfromtxt(train_data_file, delimiter=',')
        test_data = np.genfromtxt(test_data_file, delimiter=',')

    elif problem == 4:
        #Cancer
        num_samples = 5000 
        problem_type = 'classification'
        topology = [9, 12, 2]
        problem_name = 'Cancer'
        #path = 'results/Cancer' + str(num_chains) + '_' + str(max_temp)

        train_data_file = '../Datasets/data_classification/Cancer/ftrain.txt'
        test_data_file = '../Datasets/data_classification/Cancer/ftest.txt'

        train_data = np.genfromtxt(train_data_file)
        test_data = np.genfromtxt(test_data_file) 



    elif problem == 5: #Bank additional
            data = np.genfromtxt('../Datasets/data_classification/Bank/bank-processed.csv',delimiter=';')
            classes = data[:,20].reshape(data.shape[0],1)
            features = data[:,0:20]
            separate_flag = True
            problem_name = "bank-additional"

            problem_type = 'classification'
            hidden = 50
            ip = 20 #input
            output = 2
            num_samples = 5000  

            topology = [ip, hidden, output]

    elif problem == 6: #PenDigit
            train_data = np.genfromtxt('../Datasets/data_classification/PenDigit/train.csv',delimiter=',')
            test_data = np.genfromtxt('../Datasets/data_classification/PenDigit/test.csv',delimiter=',')
            problem_name = "PenDigit"

            problem_type = 'classification'

            for k in range(16):
                mean_train = np.mean(train_data[:,k])
                dev_train = np.std(train_data[:,k]) 
                train_data[:,k] = (train_data[:,k]-mean_train)/dev_train
                mean_test = np.mean(test_data[:,k])
                dev_test = np.std(test_data[:,k]) 
                test_data[:,k] = (test_data[:,k]-mean_test)/dev_test
            ip = 16
            hidden = 30
            output = 10

            num_samples = 5000  

            topology = [ip, hidden, output]

    elif problem == 7: #Chess
            data  = np.genfromtxt('../Datasets/data_classification/Chess/chess.csv',delimiter=';')
            classes = data[:,6].reshape(data.shape[0],1)
            features = data[:,0:6]
            separate_flag = True
            problem_name = "chess" 
            problem_type = 'classification'
            hidden = 25
            ip = 6 #input
            output = 18

            num_samples = 5000 

            topology = [ip, hidden, output]


    if separate_flag is True:
            #Normalizing Data
            for k in range(ip):
                mean = np.mean(features[:,k])
                dev = np.std(features[:,k])
                features[:,k] = (features[:,k]-mean)/dev
            train_ratio = 0.7 #Choosable
            indices = np.random.permutation(features.shape[0])
            train_data = np.hstack([features[indices[:np.int(train_ratio*features.shape[0])],:],classes[indices[:np.int(train_ratio*features.shape[0])],:]])
            test_data = np.hstack([features[indices[np.int(train_ratio*features.shape[0])]:,:],classes[indices[np.int(train_ratio*features.shape[0])]:,:]])
 

 

    problemfolder_db = 'Results_/'  # save main results

    run_nb = 0
    while os.path.exists( problemfolder_db+problem_name+'_%s' % (run_nb)):
        run_nb += 1
    if not os.path.exists( problemfolder_db+problem_name+'_%s' % (run_nb)):
        os.makedirs(  problemfolder_db+problem_name+'_%s' % (run_nb))
        path_db = (problemfolder_db+ problem_name+'_%s' % (run_nb))


    print('pt initialize')
    model = EvolutionaryParallelTempering(burn_in, train_data, test_data, topology, num_chains, max_temp, num_samples, swap_interval, path_db, population_size, problem_type=problem_type)
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
