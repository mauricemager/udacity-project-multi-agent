import random
import copy
import torch
import numpy as np

from collections import namedtuple, deque

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, scale=0.1, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu    = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed  = random.seed(seed)
        self.size  = size
        self.scale = scale
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = np.ones(self.size) * self.mu

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x  = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.size)
        self.state = x + dx
        return torch.tensor(self.state * self.scale).float()


class ReplayBuffer:
    def __init__(self, size, seed=0):
        self.buffer = deque(maxlen=size)
        self.seed   = random.seed(seed)
        self.size   = size

    def push(self, state, action, reward, next_state, done):
        state       = np.expand_dims(state, 0)
        next_state  = np.expand_dims(next_state, 0)

        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size, beta):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))

        return np.concatenate(state), action, reward, np.concatenate(next_state), done, np.ones(self.size), None

    def update_priorities(self, indices, updated_priorities):
        return

    def __len__(self):
        return len(self.buffer)
    
    
def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)
        

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
