# !/usr/bin/python
from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import random
import time
import math
import random
import os
from sklearn.model_selection import train_test_split
from network_torch import Network
from g3pcx import G3PCX
import multiprocessing







class MAIN(G3PCX):
    def __init__(self, num_samples, population_size, topology, train_data, test_data, directory, problem_type='regression', max_limit=(-5), min_limit=5):
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
        #self.initialize_sampling_parameters()
        self.create_directory(directory)
        self.errorlist=[]
        G3PCX.__init__(self, population_size, self.w_size, self.max_limits, self.min_limits,self.topology,self.train_data,self.test_data)

    def fitness_function(self, x):
#        print("now at fitness func, now the wts are",x)
        fitness = self.neural_network.evaluate_fitness(x)
        return fitness

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
    def calculate_rmse(actual, targets):
        return np.sqrt((np.square(np.subtract(np.absolute(actual), np.absolute(targets)))).mean())

    
    

    def sampler(self, save_knowledge=True):
        train_rmse_file = open(self.directory+'/train_rmse1.csv', 'w')
        test_rmse_file = open(self.directory+'/test_rmse1.csv', 'w')
        weights_initial = np.random.uniform(-5, 5, self.w_size)

        # ------------------- initialize MCMC
        self.start_time = time.time()

        train_size = self.train_data.shape[0]
        test_size = self.test_data.shape[0]
        y_test = self.test_data[:,4]
        y_train = self.train_data[:,4]
        weights_current = weights_initial.copy()
        # weights_proposal = weights_initial.copy()
#        print("weights",weights_current)
        prediction_train = self.neural_network.generate_output(self.train_data, weights_current)
        prediction_test = self.neural_network.generate_output(self.test_data, weights_current)
        # eta = np.log(np.var(prediction_train - y_train))
        # tau_proposal = np.exp(eta)
        # prior = self.prior_function(weights_current, tau_proposal)
        # [likelihood, rmse_train] = self.likelihood_function(self.neural_network, self.train_data, weights_current, tau_proposal)
        rmse_test = self.calculate_rmse(prediction_test, y_test)
        rmse_train=self.calculate_rmse(prediction_train, y_train)

        # save values into previous variables
        rmse_train_current = rmse_train
        

        tempfit = 0
        self.evaluate()
        tempfit = self.fitness[self.best_index]

        # start sampling
        for sample in range(self.num_samples - 1):
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
            prediction_train = self.neural_network.generate_output(self.train_data, weights_proposal)
            rmse_train=self.calculate_rmse(prediction_train, y_train)
            self.errorlist.append(rmse_train)
            # eta_proposal = eta + np.random.normal(0, self.eta_stepsize, 1)
            # tau_proposal = np.exp(eta_proposal)
            # accept, rmse_train, rmse_test, likelihood, prior = self.evaluate_proposal(self.neural_network, self.train_data, self.test_data, weights_proposal, tau_proposal, likelihood, prior)
            
            weights_current = weights_proposal
            rmse_train_current = rmse_train
            np.savetxt(train_rmse_file, [rmse_train_current])
            if sample%1000 == 0:
                print("Sample no.=",sample)
                print("Error is =",rmse_train)
           

# =============================================================================
#             elapsed_time = ":".join(MCMC.convert_time(time.time() - self.start))
# 
#             print("Sample: {}, Best Fitness: {}, Proposal: {}, Time Elapsed: {}".format(sample, rmse_train_current, rmse_train, elapsed_time))
# 
#         elapsed_time = time.time() - self.start
#         accept_ratio = num_accept/num_samples
# =============================================================================

        # Close the files
#        rmse_tr=np.loadtxt("/synthetic/train_rmse1.csv")
#        rmse_tst=np.loadtxt("test_rmse1.csv")
        
        np.savetxt("error.csv",self.errorlist)
        x_tr = np.linspace(0, self.num_samples, len(self.errorlist))
        
        
        plt.plot(x_tr, self.errorlist, label='rmse_train')
        plt.ylabel('RMSE')
        plt.legend()
        plt.savefig(('rmse_train'+str(self.population_size)+"_"+str(self.num_samples)+str(time.time())+'.png'))
        plt.clf()
        
# =============================================================================
#         
#         plt.plot(x, rmse_tst, label='rmse_test')
#         plt.ylabel('RMSE')
#         plt.legend()
#         plt.savefig(('rmse_test.png'))
#         plt.clf()
#         
# =============================================================================
        
        train_rmse_file.close()
        test_rmse_file.close()

        

if __name__ == '__main__':
    num_samples = 10000
    population_size = 400
    problem_type = 'regression'
    topology = [4, 16, 1]
    problem_name = 'synthetic'

    train_data_file = 'sunspots.csv'
    test_data_file = 'sunspots.csv'
    table=np.loadtxt("sunspots.csv")
    X = table[:,:4] 
    y = table[:,4]
    X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.33, random_state=42)
    print(X_train.shape,np.reshape(y_train,(X_train.shape[0],1)).shape)
    train_data = np.concatenate((X_train,np.reshape(y_train,(X_train.shape[0],1))),axis=1)
    test_data = np.concatenate((X_test,np.reshape(y_test,(X_test.shape[0],1))),axis=1)
    # train_data=X_traom

    model = MAIN(num_samples, population_size, topology, train_data, test_data, directory=problem_name)
    model.sampler()
    
