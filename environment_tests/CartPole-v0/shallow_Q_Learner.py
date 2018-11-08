import random

import gym
import numpy as np
import torch
from torch.autograd import Variable

from function_approxumator.perceptron import SLP
from utils.decay_schedule import LinearDecaySchedule

env = gym.make('CartPole-v0')
MAX_NUM_EPISODES = 10000
MAX_STEPS_PER_EPISODE = 300


class Shallow_Q_Learner(object):
    def __init__(self,
                 state_shape,
                 action_shape,
                 learning_rate=0.005,
                 gamma=0.98):
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.learning_rate = learning_rate
        self.Q = SLP(state_shape, action_shape)
        self.Q_optimizer = torch.optim.Adam(self.Q.parameters(), lr=1e-3)
        self.policy = self.epsilon_greedy_Q
        self.epsilon_max = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = LinearDecaySchedule(initial_value=self.epsilon_max,
                                                 final_value=self.epsilon_min,
                                                 max_steps=0.5*MAX_NUM_EPISODES * MAX_STEPS_PER_EPISODE)
        self.step_num = 0

    def get_action(self, observation):
        return self.policy(observation)

    def epsilon_greedy_Q(self, observation):
        if random.random() < self.epsilon_decay(self.step_num):
            action = random.choice([i for i in range(self.action_shape)])
        else:
            action = np.argmax(self.Q(observation).data.numpy())
        return action
