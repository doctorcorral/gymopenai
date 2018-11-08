import random

import gym
import numpy as np
import torch
from torch.autograd import Variable

from function_approxumator.perceptron import SLP
from utils.decay_schedule import LinearDecaySchedule
