"""
:-------------------------------------------------------------------------------------------------------------------:
: Title         : Distributed Neuroevolutionry Reinforcement Learning Algorithms for Automatic Trading                                    :
: Author        : Rishabh Gupta (rishabhgupta05@gmail.com), Dr Rohitash Chandra (rohitash.chandra@sydney.edu.au)     :
: Organisation  : Centre for Translational Data Science                                                             :
:-------------------------------------------------------------------------------------------------------------------:
"""
 
from __future__ import division
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import random
import time
import math
import random
import os, sys
from pprint import pprint

from config import opt
from network import Network
from g3pcx import G3PCX


class Replica(G3PCX, multiprocessing.Process):
    def __init__(self, num_samples, burn_in, population_size, topology, train_data, test_data, directory, temperature, swap_interval, parameter_queue, problem_type,  main_process, event, max_limit=(-5), min_limit=5):
        # MULTIPROCESSING CLASS CONSTRUCTOR
        multiprocessing.Process.__init__(self)
        self.process_id = temperature
        self.parameter_queue = parameter_queue
        self.signal_main = main_process
        self.event =  event
        #PARALLEL COMPUTING VARIABLES
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
    def multinomial_likelihood(neural_network, data, weights):
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
        return [rmse, accuracy]

    @staticmethod
    def classification_prior(sigma_squared, weights):
        part_1 = -1 * ((weights.shape[0]) / 2) * np.log(sigma_squared)
        part_2 = 1 / (2 * sigma_squared) * (sum(np.square(weights)))
        log_loss = part_1 - part_2
        return log_loss

    @staticmethod
    def gaussian_likelihood(neural_network, data, weights):
        desired = data[:, neural_network.topology[0]: neural_network.topology[0] + neural_network.topology[2]]
        prediction = neural_network.generate_output(data, weights)
        rmse = Network.calculate_rmse(prediction, desired)
#        loss = -0.5 * np.log(2 * np.pi * tausq) - 0.5 * np.square(desired - prediction) / tausq
        return [rmse]

    @staticmethod
    def gaussian_prior(sigma_squared, nu_1, nu_2, weights, tausq):
        part1 = -1 * (weights.shape[0] / 2) * np.log(sigma_squared)
        part2 = 1 / (2 * sigma_squared) * (sum(np.square(weights)))
        log_loss = part1 - part2 - (1 + nu_1) * np.log(tausq) - (nu_2 / tausq)
        return log_loss

    def likelihood_function(self, neural_network, data, weights):
        if self.problem_type == 'regression':
            rmse = self.gaussian_likelihood(neural_network, data, weights)
            return  rmse, None
        elif self.problem_type == 'classification':
            rmse, accuracy = self.multinomial_likelihood(neural_network, data, weights)
            return  rmse, accuracy

    def prior_function(self, weights, tau):
        if self.problem_type == 'regression':
            loss = self.gaussian_prior(self.sigma_squared, self.nu_1, self.nu_2, weights, tau)
        elif self.problem_type == 'classification':
            loss = self.classification_prior(self.sigma_squared, weights)
        # quit()
        return loss

    def evaluate_proposal(self, neural_network, train_data, test_data, weights_proposal):
        accept = True
        
        
        rmse_test_proposal, acc_test = self.likelihood_function(neural_network, test_data, weights_proposal)
        rmse_train_proposal, acc_train = self.likelihood_function(neural_network, train_data, weights_proposal)
# =============================================================================
#         prior_proposal = self.prior_function(weights_proposal, tau_proposal)
#         difference_likelihood = likelihood_proposal - likelihood_current
#         difference_prior = prior_proposal - prior_current
#         mh_ratio = min(1, np.exp(min(709, difference_likelihood)))
#         u = np.random.uniform(0,1)
#         if u < mh_ratio:
#             accept = True
#             likelihood_current = likelihood_proposal
#             prior_proposal = prior_current
# =============================================================================
        if acc_train == None:
            return accept, rmse_train_proposal, rmse_test_proposal
        else:
            return accept, rmse_train_proposal, rmse_test_proposal,acc_train,acc_test


    def run(self):
        save_knowledge = True
        train_rmse_file = open(os.path.join(self.directory, 'train_rmse_{:.4f}.csv'.format(self.temperature)), 'w')
        test_rmse_file = open(os.path.join(self.directory, 'test_rmse_{:.4f}.csv'.format(self.temperature)), 'w')
        if self.problem_type == 'classification':
            train_acc_file = open(os.path.join(self.directory, 'train_acc_{:.4f}.csv'.format(self.temperature)), 'w')
            test_acc_file = open(os.path.join(self.directory, 'test_acc_{:.4f}.csv'.format(self.temperature)), 'w')
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
# =============================================================================
#         eta = np.log(np.var(prediction_train - y_train))
#         tau_proposal = np.exp(eta)
#         prior = self.prior_function(weights_current, tau_proposal)
#         [likelihood, rmse_train, acc_train] = self.likelihood_function(self.neural_network, self.train_data, weights_current, tau_proposal, self.temperature)
# 
# =============================================================================
        [rmse_train, acc_train] = self.likelihood_function(self.neural_network, self.train_data, weights_current)
#        print("RMSE Train=",rmse_train)
        rmse_test = Network.calculate_rmse(prediction_test, y_test)
        if self.problem_type == 'classification':
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
            if self.problem_type == 'classification':
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
                if save_knowledge:
                    train_rmse_file.write(str(rmse_train_current)+"\n")
                    test_rmse_file.write(str(rmse_test_current)+"\n")
                    if self.problem_type == 'classification':
                        train_acc_file.write(str(acc_train_current)+"\n")
                        test_acc_file.write(str(acc_test_current)+"\n")
                        writ += 1
                continue
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
# =============================================================================
#             eta_proposal = eta + np.random.normal(0, self.eta_stepsize, 1)
#             tau_proposal = np.exp(eta_proposal)
# =============================================================================
            if self.problem_type == 'classification':
#                accept, rmse_train, rmse_test, acc_train, acc_test, likelihood, prior = self.evaluate_proposal(self.neural_network, self.train_data, self.test_data, weights_proposal, tau_proposal, likelihood, prior)
                accept, rmse_train, rmse_test, acc_train, acc_test= self.evaluate_proposal(self.neural_network, self.train_data, self.test_data, weights_proposal)
            else:
                accept, rmse_train, rmse_test = self.evaluate_proposal(self.neural_network, self.train_data, self.test_data, weights_proposal)

#            print(f"Rmse train is {rmse_train}")
            if accept:
                num_accept += 1
                weights_current = weights_proposal
#                eta = eta_proposal
                # save values into previous variables
                rmse_train_current = rmse_train
                rmse_test_current = rmse_test
                if self.problem_type == 'classification':
#                    print(f"Acc Train= {acc_train} with chain num= {self.temperature}")
                    acc_train_current = acc_train
                    acc_test_current = acc_test

            if save_knowledge:
                train_rmse_file.write(str(rmse_train_current)+"\n")
                test_rmse_file.write(str(rmse_test_current)+"\n")
                if self.problem_type == 'classification':
                    print(f"Acc Train= {acc_train} with chain num= {self.temperature}")
                    train_acc_file.write(str(acc_train_current)+"\n")
                    test_acc_file.write(str(acc_test_current)+"\n")
                    writ += 1

            #SWAPPING PREP
            if (sample % self.swap_interval == 0 and sample != 0 ):
                # print('\nTemperature: {} Swapping weights: {}'.format(self.temperature, weights_current[:2]))
#                param = np.concatenate([weights_current, np.asarray([eta]).reshape(1), np.asarray([likelihood*self.temperature]),np.asarray([self.temperature])])
                param=weights_current
                self.parameter_queue.put(param)#parameter queue contains the multiprocessing chains  
                self.signal_main.set() #it is the main process with the Event "wait chain"
                #the signal_main basically gives a call to the main process to stop the main chain so that
                #the swap can be performed. The packet of information of weights to swap is ready.
                
                self.event.wait()#it is the Event chain which perform the main chain wait
                # retrieve parameters fom queues if it has been swapped
                # retrieve parameters fom queues if it has been swapped
                if not self.parameter_queue.empty() :
                    try:
                        result =  self.parameter_queue.get()
                        #print(self.temperature, w, 'param after swap')
                        weights_current = result[0:self.w_size]
                        self.population[self.best_index] = weights_current
                        self.fitness[self.best_index] = self.fitness_function(weights_current)
# =============================================================================
#                         eta = result[self.w_size]
#                         likelihood = result[self.w_size+1]/self.temperature
# =============================================================================
                        # likelihood = self.likelihood_function(self.neural_network, self.train_data, weights_current, np.exp(eta))
#                        print('Temperature: {} Swapped weights: {}'.format(self.temperature, weights_current[:2]))
                    except:
                        print ('error')
                else:
                    print("Khali")
                self.event.clear()
            elapsed_time = ":".join(Replica.convert_time(time.time() - self.start_time))

#            print("Temperature: {:.2f} Sample: {:d}, Best Fitness: {:.4f}, Proposal: {:.4f}, Time Elapsed: {:s}".format(self.temperature, sample, rmse_train_current, rmse_train, elapsed_time))

        elapsed_time = time.time() - self.start_time
        print(f"Elapsed time is {elapsed_time}")
# =============================================================================
#         accept_ratio = num_accept/self.num_samples
# =============================================================================
        print("Written {} values for Accuracies".format(writ))
        # Close the files
        train_rmse_file.close()
        test_rmse_file.close()
#        print("Temperature: {} done, {} samples sampled out of {}".format(self.temperature, sample, self.num_samples))

class EvoPL(object):

    def __init__(self, opt, path, geometric=True):
        #FNN Chain variables
        self.opt = opt
        self.train_data = opt.train_data
        self.test_data = opt.test_data
        self.topology = list(map(int,opt.topology.split(',')))
        self.num_param = (self.topology[0] * self.topology[1]) + (self.topology[1] * self.topology[2]) + self.topology[1] + self.topology[2]
        self.problem_type = opt.problem_type
        #Parallel Tempering variables
        self.burn_in = opt.burn_in
        self.swap_interval = opt.swap_interval
        self.path = path
        self.max_temp = opt.max_temp
        self.num_swap = 0
        self.total_swap_proposals = 0
        self.num_chains = opt.num_chains
        self.chains = []
        self.temperatures = []
        self.num_samples = int(opt.num_samples/self.num_chains)
        self.geometric = geometric
        self.population_size = opt.population_size
        # create queues for transfer of parameters between process chain
        self.parameter_queue = [multiprocessing.Queue() for i in range(self.num_chains)]
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
            for i in range(0,self.num_chains):
                self.temperatures.append(i)
#            betas = self.default_beta_ladder(2, ntemps=self.num_chains, Tmax=self.max_temp)
#            self.temperatures = [np.inf if beta == 0 else 1.0/beta for beta in betas]
        #Linear Spacing
        else:
            temp = 2
            for i in range(0,self.num_chains):
                self.temperatures.append(temp)
                temp += 2.5 #(self.maxtemp/self.num_chains)
                print (self.temperatures[i])

    def initialize_chains(self):
        self.assign_temperatures()
#        weights = np.random.randn(self.num_param)
        for chain in range(0, self.num_chains):
            self.chains.append(Replica(self.num_samples, self.burn_in, self.population_size, self.topology, self.train_data, self.test_data, self.path, self.temperatures[chain], self.swap_interval, self.parameter_queue[chain], self.problem_type, main_process=self.wait_chain[chain], event=self.event[chain]))

    def swap_procedure(self, parameter_queue_1, parameter_queue_2):
        if not parameter_queue_2.empty() and not parameter_queue_1.empty():
            param_1 = parameter_queue_1.get()
            param_2 = parameter_queue_2.get()
            swapped = True
            param_temp =  param_1
            param_1 = param_2
            param_2 = param_temp
            print("Swapped {}, {}".format(param_1[:2], param_2[:2]))
            return param_1, param_2, swapped
# =============================================================================
#             w_1 = param_1[0:self.num_param]
#             eta_1 = param_1[self.num_param]
#             likelihood_1 = param_1[self.num_param+1]
#             T_1 = param_1[self.num_param+2]
#             w_2 = param_2[0:self.num_param]
#             eta_2 = param_2[self.num_param]
#             likelihood_2 = param_2[self.num_param+1]
#             T_2 = param_2[self.num_param+2]
# =============================================================================
            #SWAPPING PROBABILITIES
# =============================================================================
#             try:
#                 swap_proposal =  min(1,0.5*np.exp(likelihood_2 - likelihood_1))
#             except OverflowError:
#                 swap_proposal = 1
# =============================================================================
# =============================================================================
#             u = np.random.uniform(0,1)
#             if u < swap_proposal:
# =============================================================================
# =============================================================================
#             swapped = True
#                 self.num_swap += 1
#                 param_temp =  param_1
#                 param_1 = param_2
#                 param_2 = param_temp
#                 print("Swapped {}, {}".format(param_1[:2], param_2[:2]))
#             else:
#                 print("No swapping!!")
#                 swapped = False
#             self.total_swap_proposals += 1
# =============================================================================
# =============================================================================
#             return param_1, param_2, swapped
# =============================================================================
        else:
            print("No Swapping occured")
            self.total_swap_proposals += 1
            raise Exception('empty queue')
            return

    def run_chains(self):
        x_test = np.linspace(0,1,num=self.test_data.shape[0])
        x_train = np.linspace(0,1,num=self.train_data.shape[0])
        # only adjacent chains can be swapped therefore, the number of proposals is ONE less num_chains
        swap_proposal = np.ones(self.num_chains-1)
        # create parameter holders for paramaters that will be swapped
        replica_param = np.zeros((self.num_chains, self.num_param))
# =============================================================================
#         likelihood = np.zeros(self.num_chains)
#         eta = np.zeros(self.num_chains)
# =============================================================================
        # Define the starting and ending of MCMC Chains
        start = 0
        end = self.num_samples-1
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
                    print("Temp {} Dead".format(self.chains[index].temperature))

            if count == self.num_chains:
                break
            print("Waiting")
            timeout_count = 0
            for index in range(0,self.num_chains):
                print("Waiting for chain: {}".format(index+1))
                flag = self.wait_chain[index].wait(timeout=2)
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
            print("Chain {} joined".format(index))
        self.chain_queue.join()
        print("Done")

        #GETTING DATA
        burn_in = int(self.num_samples*self.burn_in)
        rmse_train = np.zeros((self.num_chains,self.num_samples - burn_in))
        rmse_test = np.zeros((self.num_chains,self.num_samples - burn_in))
        if self.problem_type == 'classification':
            acc_train = np.zeros((self.num_chains,self.num_samples - burn_in))
            acc_test = np.zeros((self.num_chains,self.num_samples - burn_in))
        accept_ratio = np.zeros((self.num_chains,1))

        for i in range(self.num_chains):

            file_name = os.path.join(self.path, 'test_rmse_{:.4f}.csv'.format(self.temperatures[i]))
            dat = np.genfromtxt(file_name, delimiter=',')
            rmse_test[i,:] = dat[burn_in:]

            file_name = os.path.join(self.path, 'train_rmse_{:.4f}.csv'.format(self.temperatures[i]))
            dat = np.genfromtxt(file_name, delimiter=',')
            rmse_train[i,:] = dat[burn_in:]

            if self.problem_type == 'classification':
                file_name = os.path.join(self.path, 'test_acc_{:.4f}.csv'.format(self.temperatures[i]))
                dat = np.genfromtxt(file_name, delimiter=',')
                acc_test[i,:] = dat[burn_in:]

                file_name = os.path.join(self.path, 'train_acc_{:.4f}.csv'.format(self.temperatures[i]))
                dat = np.genfromtxt(file_name, delimiter=',')
                acc_train[i,:] = dat[burn_in:]

        self.rmse_train = rmse_train.reshape(self.num_chains*(self.num_samples - burn_in), 1)
        self.rmse_test = rmse_test.reshape(self.num_chains*(self.num_samples - burn_in), 1)
        if self.problem_type == 'classification':
            self.acc_train = acc_train.reshape(self.num_chains*(self.num_samples - burn_in), 1)
            self.acc_test = acc_test.reshape(self.num_chains*(self.num_samples - burn_in), 1)

        # PLOT THE RESULTS
        self.plot_figures()

        print("NUMBER OF SWAPS MAIN =", total_swaps_main)
        print("SWAP ACCEPTANCE = ", self.num_swap*100/self.total_swap_proposals," %")
        print("SWAP ACCEPTANCE MAIN = ", swaps_appected_main*100/total_swaps_main," %")

        if self.problem_type == 'classification':
            return (self.rmse_train, self.rmse_test, self.acc_train, self.acc_test)
        return (self.rmse_train, self.rmse_test)



    def plot_figures(self):
        
        # X-AXIS 
        x = np.linspace(0, 1, len(self.rmse_train))

        # PLOT TRAIN RMSE
        plt.plot(x, self.rmse_train, label='rmse_train')
        plt.xlabel('samples')
        plt.ylabel('RMSE')
        plt.title('{} RMSE Train'.format(self.opt.problem.upper()))
        plt.legend()
        plt.savefig(os.path.join(self.path, 'rmse_train.png'))
        plt.clf()

        # PLOT TEST RMSE
        plt.plot(x, self.rmse_test, label='rmse_test')
        plt.xlabel('samples')
        plt.ylabel('RMSE')
        plt.title('{} RMSE Test'.format(self.opt.problem.upper()))
        plt.legend()
        plt.savefig(os.path.join(self.path, 'rmse_test.png'))
        plt.clf()

        if self.problem_type == 'classification':
            # PLOT TRAIN ACCURACY
            plt.plot(x, self.acc_train, label='acc_train')
            plt.xlabel('samples')
            plt.ylabel('Accuracy %')
            plt.title('{} Accurace Train'.format(self.opt.problem.upper()))
            plt.legend()
            plt.savefig(os.path.join(self.path, 'acc_train.png'))
            plt.clf()

            # PLOT TEST ACCURACY
            plt.plot(x, self.acc_test, label='acc_test')
            plt.xlabel('samples')
            plt.ylabel('Accuracy %')
            plt.title('{} Accurace Test'.format(self.opt.problem.upper()))
            plt.legend()
            plt.savefig(os.path.join(self.path, 'acc_test.png'))
            plt.clf()

def make_directory(path):
    if not os.path.isdir(path):
        os.makedirs(path)

if __name__ == '__main__':

    # CREATE RESULTS DIRECTORY
    make_directory('Results')
    results_dir = os.path.join('Results', '{}_{}'.format(opt.problem, opt.run_id))
    make_directory(results_dir)

    logfile = os.path.join(results_dir, 'log.txt')
    with open(logfile, 'w') as stream:
        
        stream.write(str(opt))
        
    print("opt is",opt)

    # READ DATA
    data_path = os.path.join(opt.root, 'Datasets')
    opt.train_data = np.genfromtxt(os.path.join(data_path, opt.train_data), delimiter=',')
    opt.test_data = np.genfromtxt(os.path.join(data_path, opt.test_data), delimiter=',')


    # CREATE EVOLUTIONARY PT CLASS
    evo_pt = EvoPL(opt, results_dir)
    
    # INITIALIZE PT CHAINS
    evo_pt.initialize_chains()

    # RUN EVO PT
    if opt.problem_type == 'classification':
        rmse_train, rmse_test, acc_train, acc_test = evo_pt.run_chains()
    elif opt.problem_type == 'regression':
        rmse_train, rmse_test = evo_pt.run_chains()
