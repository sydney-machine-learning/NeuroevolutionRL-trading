import gym
import math
import numpy as np
from collections import deque
import matplotlib.pyplot as plt



import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

env = gym.make('CartPole-v0')
env.seed(101)
np.random.seed(101)

print('observation space:', env.observation_space)
print('action space:', env.action_space)
# print('  - low:', env.action_space.low)
# print('  - high:', env.action_space.high)

class Agent(nn.Module):
    def __init__(self, env, h_size=16):
        super(Agent, self).__init__()
        self.env = env
        # state, hidden layer, action sizes
        self.s_size = env.observation_space.shape[0]
        self.h_size = h_size
        self.a_size = 2
        # define layers
        self.fc1 = nn.Linear(self.s_size, self.h_size)
        self.fc2 = nn.Linear(self.h_size, self.a_size)
        
    def set_weights(self, weights):
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
        self.fc1.bias.data.copy_(fc1_b.view_as(self.fc1.bias.data))
        self.fc2.weight.data.copy_(fc2_W.view_as(self.fc2.weight.data))
        self.fc2.bias.data.copy_(fc2_b.view_as(self.fc2.bias.data))
    
    def get_weights_dim(self):
        return (self.s_size+1)*self.h_size + (self.h_size+1)*self.a_size
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        return x.cpu().data
        
    def evaluate(self, weights, gamma=1.0, max_t=5000):
        self.set_weights(weights)
        episode_return = 0.0
        state = self.env.reset()
        for t in range(max_t):
            state = torch.from_numpy(state).float().to(device)
            action=self.forward(state)
            values,indices=torch.max(action,0)
            action=indices.item()
            
            state, reward, done, _ = self.env.step(action)
            episode_return += reward * math.pow(gamma, t)
            if done:
                break
        return episode_return
    
agent = Agent(env).to(device)





 




def prior_likelihood(sigma_squared, nu_1, nu_2, w, tausq,h,d):
    h = h  # number hidden neurons
    d = d  # number input neurons
    part1 = -1 * ((d * h + h + 2) / 2) * np.log(sigma_squared)
    part2 = 1 / (2 * sigma_squared) * (sum(np.square(w)))
    log_loss = part1 - part2 - (1 + nu_1) * np.log(tausq) - (nu_2 / tausq)
    return log_loss

def likelihood_func(sigma,gamma, max_t,w, tausq,n_iterations,norm_rew):
        #y = data[:, self.topology[0]]
        
        rmse = norm_rew.mean()
        loss = -0.5 * np.log(2 * math.pi * tausq) - 0.5 * np.square(norm_rew) / tausq
        return [np.sum(loss), rmse]




def cem(n_iterations=1000, max_t=50, gamma=1, print_every=10, pop_size=50, elite_frac=0.3, sigma=0.5):
    """PyTorch implementation of the cross-entropy method.
        
    Params
    ======
        n_iterations (int): maximum number of training iterations
        max_t (int): maximum number of timesteps per episode
        gamma (float): discount rate
        print_every (int): how often to print average score (over last 100 episodes)
        pop_size (int): size of population at each iteration
        elite_frac (float): percentage of top performers to use in update
        sigma (float): standard deviation of additive noise
    """
    n_elite=int(pop_size*elite_frac)

    scores_deque = deque(maxlen=100)
    scores = []
    
    pos_w = np.ones((n_iterations, agent.get_weights_dim()))
    pos_tau = np.ones((n_iterations, 1))

    rmse_train = np.zeros(n_iterations)
    rmse_test = np.zeros(n_iterations)



    w = sigma*np.random.randn(agent.get_weights_dim())
    w_proposal = np.random.randn(w_size)


    step_w = 0.02  # defines how much variation you need in changes to w
    step_eta = 0.01


    #now declare the nn and do aforward pass
    weights_pop = [best_weight + (sigma*np.random.randn(agent.get_weights_dim())) for i in range(n_iterations)]
    rewards = np.array([agent.evaluate(weights, gamma, max_t) for weights in weights_pop])
    eta = np.log(np.var(rewards))
    tau_pro = np.exp(eta)

    sigma_squared = 25
    nu_1 = 0
    nu_2 = 0
    #basically rewards here is the loss



    prior_likelihood = prior_likelihood(sigma_squared, nu_1, nu_2, w, tau_pro)  # takes care of the gradients
    [likelihood, pred_train, rmsetrain] = likelihood_func(neuralnet, self.train_x,self.train_y, w, tau_pro)
    







    

    for i_iteration in range(1, n_iterations+1):
        weights_pop = [best_weight + (sigma*np.random.randn(agent.get_weights_dim())) for i in range(pop_size)]
        rewards = np.array([agent.evaluate(weights, gamma, max_t) for weights in weights_pop])
        
        
        
        [pos_w, pos_tau, fx_train, fx_test, x_train, x_test, rmse_train, rmse_test, accept_ratio] = mcmc.sampler()
        print ('sucessfully sampled')


        elite_idxs = rewards.argsort()[-n_elite:]
        elite_weights = [weights_pop[i] for i in elite_idxs]
        best_weight = np.array(elite_weights).mean(axis=0)

        reward = agent.evaluate(best_weight, gamma=1.0)
        scores_deque.append(reward)
        scores.append(reward)
        
        torch.save(agent.state_dict(), 'checkpoint.pth')
        
        if i_iteration % print_every == 0:
            print('Episode {}\tAverage Score: {:.2f}'.format(i_iteration, np.mean(scores_deque)))

        if np.mean(scores_deque)>=90.0:
            print('\nEnvironment solved in {:d} iterations!\tAverage Score: {:.2f}'.format(i_iteration-100, np.mean(scores_deque)))
            break
    return scores

scores = cem()

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores)+1), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()





# load the weights from file
agent.load_state_dict(torch.load('checkpoint.pth'))

state = env.reset()
while True:
    state = torch.from_numpy(state).float().to(device)
    with torch.no_grad():
        action = agent(state)
        
    env.render()
    action = (round(((max(action))).item()))
    next_state, reward, done, _ = env.step(action)
    state = next_state
    if done:
        break

env.close()
