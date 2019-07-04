
import numpy as np
import time
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from sklearn import datasets


class TheModelClass(nn.Module):
    def __init__(self,topology):
        super(TheModelClass, self).__init__()
        self.topology=topology
        self.s_size=topology[0]
        self.h_size=topology[1]
        self.a_size=topology[2]
        self.fc1 = nn.Linear(self.s_size, self.h_size)
        self.fc2 = nn.Linear(self.h_size, self.a_size)
        self.train_data = train_data
        self.test_data = test_data
        
	def forward(self, x):
		x = F.sigmoid(self.fc1(x))
		x = F.sigmoid(self.fc2(x))
		return x

	def get_weights_dim(self):
		return (self.s_size+1)*self.h_size + (self.h_size+1)*self.a_size

	def set_weights(self, weights):
		# print("weights",weights[0:5])
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