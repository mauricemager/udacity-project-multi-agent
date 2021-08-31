import torch
import numpy as np

from data.model import Network
from data.utilities import OUNoise, hard_update
from torch.optim import Adam



class DDPGAgent:
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.actor  = Network(config.state_size,  config.actor_hidden_sizes[0],config.actor_hidden_sizes[1], 
                              config.action_size, config.seed, actor=True).to(config.device)
        
        self.critic = Network(2 * (config.state_size + config.action_size), config.critic_hidden_sizes[0],
                              config.critic_hidden_sizes[1], 1, config.seed).to(config.device)

        self.target_actor  = Network(config.state_size,  config.actor_hidden_sizes[0], config.actor_hidden_sizes[1], 
                                     config.action_size, config.seed, actor=True).to(config.device)
        
        self.target_critic = Network(2 * (config.state_size + config.action_size), config.critic_hidden_sizes[0],
                                     config.critic_hidden_sizes[1], 1, config.seed).to(config.device)

        self.noise = OUNoise(config.action_size, seed=config.seed, scale=1.0, mu=config.mu, 
                             theta=config.theta, sigma=config.sigma)

        hard_update(self.target_actor,  self.actor)
        hard_update(self.target_critic, self.critic)

        self.actor_optimizer  = Adam(self.actor.parameters(),  lr=config.lr_actor)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=config.lr_critic,
                                     weight_decay=config.critic_weight_decay)

    def act(self, obs, noise=0.0):
        action = self.actor(obs) + noise * self.noise.sample().to(self.config.device)
        return action

    def target_act(self, obs, noise=0.0):
        action = self.target_actor(obs) + noise * self.noise.sample().to(self.config.device)
        return action

    def reset(self):
        self.noise.reset()
