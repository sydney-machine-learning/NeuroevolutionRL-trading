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
import time
import os

from config import opt
from network import Network
from g3pcx import G3PCX

import gym
env = gym.make('CartPole-v0')
env.seed(101)
np.random.seed(101)


print('observation space:', env.observation_space)
print('action space:', env.action_space)

class Replica(G3PCX, multiprocessing.Process):
    def __init__(self, num_samples, population_size, topology, directory, temperature, swap_interval, parameter_queue,Senv,main_process, event, max_limit=(-5), min_limit=5):
        # MULTIPROCESSING CLASS CONSTRUCTOR
        multiprocessing.Process.__init__(self)
        self.process_id = temperature
        self.parameter_queue = parameter_queue
        self.signal_main = main_process
        self.event =  event
        #PARALLEL COMPUTING VARIABLES
        self.temperature = temperature
        self.swap_interval = swap_interval
#      
        # RL VARIABLES
        self.num_samples = num_samples
        self.topology = topology

        self.directory = directory
        self.w_size = (topology[0] * topology[1]) + (topology[1] * topology[2]) + topology[1] + topology[2]
#        self.neural_network = Network(topology, train_data, test_data,env)
        self.neural_network = Network(topology,env)

        self.min_limits = np.repeat(min_limit, self.w_size)
        self.max_limits = np.repeat(max_limit, self.w_size)
#        self.initialize_sampling_parameters()
        self.create_directory(directory)
        G3PCX.__init__(self, population_size, self.w_size, self.max_limits, self.min_limits)

    def fitness_function(self, x):
#        print("111")
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




    def run(self):
        save_knowledge = True
        reward_file = open(os.path.join(self.directory, 'reward_{:.4f}.csv'.format(self.temperature)), 'w')

        weights_initial = np.random.uniform(-5, 5, self.w_size)
        
        # ------------------- initialize MCMC
        self.start_time = time.time()
        weights_current = weights_initial.copy()         
        reward=self.neural_network.evaluate_fitness(weights_current)
        print(f"THe first reward for chain {self.temperature} is {reward}")


        reward_current=1/reward
        num_accept = 0
        tempfit = 0
        self.evaluate()
        tempfit = self.fitness[self.best_index]
        writ = 0
        if save_knowledge:
            reward_file.write(str(reward_current)+"\n")

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
                    reward_file.write(str(reward_current)+"\n")
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
           
            weights_proposal = self.population[self.best_index]

            reward=self.neural_network.evaluate_fitness(weights_proposal)
            accept=True
            if accept:
                num_accept += 1
                weights_current = weights_proposal
                reward_current =1/ reward

            if save_knowledge:
                print(f"reward is {reward_current} for chain {self.temperature}")
                reward_file.write(str(reward_current)+"\n")


            #SWAPPING PREP
            if (sample % self.swap_interval == 0 and sample != 0 ):

                param=weights_current
                self.parameter_queue.put(param)#parameter queue contains the multiprocessing chains  
                self.signal_main.set() #it is the main process with the Event "wait chain"
                #the signal_main basically gives a call to the main process to stop the main chain so that
                #the swap can be performed. The packet of information of weights to swap is ready.
                
                self.event.wait()#it is the Event chain which perform the main chain wait
                # retrieve parameters fom queues if it has been swapped
                if not self.parameter_queue.empty() :
                    try:
                        result =  self.parameter_queue.get()
                        #print(self.temperature, w, 'param after swap')
                        weights_current = result[0:self.w_size]
                        self.population[self.best_index] = weights_current
                        self.fitness[self.best_index] = self.fitness_function(weights_current)

                    except:
                        print ('error')
                else:
                    print("Khali")
                self.event.clear()
            elapsed_time = ":".join(Replica.convert_time(time.time() - self.start_time))

            
            print(f"Temperature: {self.temperature} Sample:{sample},Best Reaward: {reward_current}, Time Elapsed: {elapsed_time} Chain{self.temperature}")
            
           

        elapsed_time = time.time() - self.start_time
        print(f"Elapsed time is {elapsed_time}")
        print("Written {} values for Accuracies".format(writ))
        # Close the files
        reward_file.close()


class EvoPL(object):

    def __init__(self, opt, path,env, geometric=True):
        #RL variables
        
        self.opt = opt
        self.env=env
        self.s_size=env.observation_space.shape[0]
        self.h_size=16
        self.a_size= env.action_space.n
        

        self.topology = [self.s_size,self.h_size,self.a_size]
        self.num_param = (self.topology[0] * self.topology[1]) + (self.topology[1] * self.topology[2]) + self.topology[1] + self.topology[2]
        #Parallel Computing variables
        self.swap_interval = opt.swap_interval
        self.path = path
        self.num_swap = 0
        self.total_swap_proposals = 0
        self.num_chains = 1
        self.chains = []
        self.temperatures = []
        self.num_samples= 100
        self.geometric = geometric
        self.population_size= 100
        # create queues for transfer of parameters between process chain
        self.parameter_queue = [multiprocessing.Queue() for i in range(self.num_chains)]
        self.chain_queue = multiprocessing.JoinableQueue()
        self.wait_chain = [multiprocessing.Event() for i in range (self.num_chains)]
        self.event = [multiprocessing.Event() for i in range (self.num_chains)]
        make_directory(path)


    

    def assign_temperatures(self):
        for i in range(0,self.num_chains):
            print(f"temp is{i}")
            self.temperatures.append(i)

    def initialize_chains(self):
        self.assign_temperatures()
        for chain in range(0, self.num_chains):
            self.chains.append(Replica(self.num_samples, self.population_size, self.topology, self.path, self.temperatures[chain], self.swap_interval, self.parameter_queue[chain], self.env,main_process=self.wait_chain[chain], event=self.event[chain]))

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

        else:
            print("No Swapping occured")
            self.total_swap_proposals += 1
            raise Exception('empty queue')
            return

    def run_chains(self):
        # Define the starting and ending of RL Chains
        start = 0
        end = self.num_samples-1
        #RUN  CHAINS
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
        reward= np.zeros((self.num_chains,self.num_samples))
        for i in range(self.num_chains):
            file_name = os.path.join(self.path, 'reward_{:.4f}.csv'.format(self.temperatures[i]))
            dat = np.genfromtxt(file_name, delimiter=',')
            reward[i,:] = dat
        self.rewards = reward.reshape(self.num_chains*(self.num_samples), 1)   #this was line 684 earlier
        # PLOT THE RESULTS
        self.plot_figures()
        print("NUMBER OF SWAPS MAIN =", total_swaps_main)
        return self.rewards



    def plot_figures(self):
        
        # X-AXIS 
        x = np.linspace(0, 1, len(self.rewards))

        # PLOT TRAIN RMSE
        plt.plot(x, self.rewards, label='rewards')
        plt.xlabel('samples')
        plt.ylabel('Reward')
        plt.title('Reward as training')
        plt.legend()
        plt.savefig(os.path.join(self.path, 'rewards.png'))
        plt.clf()

def make_directory(path):
    if not os.path.isdir(path):
        os.makedirs(path)

if __name__ == '__main__':

    # CREATE RESULTS DIRECTORY
    make_directory('Results')
    results_dir = os.path.join('Results', '{}'.format(opt.run_id))
    make_directory(results_dir)

    logfile = os.path.join(results_dir, 'log.txt')
    with open(logfile, 'w') as stream:
        
        stream.write(str(opt))
        
    print("opt is",opt)


    # CREATE EVOLUTIONARY DISTRIBUTED RL CLASS
    evo_pt = EvoPL(opt, results_dir,env)
    
    # INITIALIZE EVO RL CHAINS
    evo_pt.initialize_chains()
    rewards = evo_pt.run_chains()
