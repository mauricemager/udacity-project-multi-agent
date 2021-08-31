import torch
import numpy as np

from data.ddpg import DDPGAgent
from data.utilities import ReplayBuffer, soft_update



class MADDPG():
    """MADDPG Algorithm class"""
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.seed   = config.seed 
        self.agents = [DDPGAgent(self.config) for _ in range(config.num_agents)]
        self.iter   = 0 
        
        self.learn_iter    = 0
        self.beta_function = lambda x: min(1.0, self.config.beta + x * (1.0 - self.config.beta) / self.config.beta_decay)
        self.memory        = ReplayBuffer(self.config.buffer_size, self.config.seed)
    
    
    def reset(self):
        """reset all agent noise parameters"""
        
        for ddpg_agent in self.agents:
            ddpg_agent.reset()

            
    def act(self, obs_all_agents, noise=0.0):
        """get actions from all agents in the MADDPG object"""
        
        obs_all_agents = torch.tensor(obs_all_agents, dtype=torch.float).to(self.config.device)
        actions = [np.clip(agent.act(obs, noise).cpu().data.numpy(), -1, 1) 
                   for agent, obs in zip(self.agents, obs_all_agents)]
        return actions
    
    
    def target_act(self, obs_all_agents, noise=0.0):
        """get target network actions from all the agents in the MADDPG object """
        
        target_actions = [ddpg_agent.target_act(obs, noise) for ddpg_agent, obs 
                          in zip(self.agents, obs_all_agents)]
        return target_actions
    
    
    def step(self, state, action, reward, next_state, done):
        """perform one step with the MADDPG algorithm"""
        
        self.memory.push(state, action, reward, next_state, done)
        self.iter += 1

        if(len(self.memory) >= self.config.batch_size) and self.iter % self.config.update_every == 0:
            beta = self.beta_function(self.learn_iter)
            for i in range(len(self.agents)):
                samples = self.memory.sample(self.config.batch_size, beta)
                self.update(samples, i)
            self.learn_iter += 1
            self.update_targets()

            
    def convert_samples(self, samples):
        """convert all sample instances to torch device"""
        
        convert = lambda x: torch.tensor(x, dtype=torch.float).to(self.config.device)

        obs, action, reward, next_obs, done, weights, idx = samples

        obs           = np.rollaxis(obs, 1)
        next_obs      = np.rollaxis(next_obs, 1)
        obs_full      = np.hstack(obs)
        next_obs_full = np.hstack(next_obs)

        obs           = convert(obs)
        obs_full      = convert(obs_full)
        action        = convert(action)
        reward        = convert(reward)
        next_obs      = convert(next_obs)
        next_obs_full = convert(next_obs_full)
        done          = convert(np.float32(done))
        weights       = convert(weights)

        return obs, obs_full, action, reward, next_obs, next_obs_full, done, idx, weights    
    
    
    def update(self, samples, agent_number):
        """update the critics and actors of all the agents """
        
        obs, obs_full, action, reward, next_obs, next_obs_full, \
            done, idx, weights = self.convert_samples(samples)      
        
        agent = self.agents[agent_number]
        agent.critic_optimizer.zero_grad()

        # critic loss = batch mean of (y- Q(s,a) from target network)^2
        # y = reward of this timestep + discount * Q(st+1,at+1) from target network
        target_actions = self.target_act(next_obs)
        target_actions = torch.cat(target_actions, dim=1).detach()
        # concatenate observations and actions for critic input
        target_critic_input = torch.cat((next_obs_full,target_actions), dim=1)
        
        with torch.no_grad(): q_next = agent.target_critic(target_critic_input)
        
        y = reward[..., agent_number].unsqueeze(1) + self.config.gamma \
                   * q_next * (1 - done[..., agent_number].unsqueeze(1))
        
        critic_input = torch.cat((obs_full, action.view(self.config.batch_size, -1)), dim=1)
        q = agent.critic(critic_input)

        huber_loss  = torch.nn.SmoothL1Loss()
        critic_loss = huber_loss(q, y.detach())
        critic_loss.backward()
        agent.critic_optimizer.step()

        # update actor network using policy gradient
        agent.actor_optimizer.zero_grad()
        # make input to agent
        # detach the other agents to save computation
        # saves some time for computing derivative
        q_input = [ self.agents[i].actor(ob) if i == agent_number \
                   else self.agents[i].actor(ob).detach()
                   for i, ob in enumerate(obs) ]
                
        q_input = torch.cat(q_input, dim=1)
        # combine all the actions and observations for input to critic
        # many of the obs are redundant, and obs[1] contains all useful information already
        q_input2 = torch.cat((obs_full, q_input), dim=1)
        
        # get the policy gradient
        actor_loss = -agent.critic(q_input2).mean()
        actor_loss.backward()
        # torch.nn.utils.clip_grad_norm_(agent.actor.parameters(),0.5)
        agent.actor_optimizer.step()

        al = actor_loss.cpu().detach().item()
        cl = critic_loss.cpu().detach().item()

    def update_targets(self):
        """soft update targets"""
        
        self.iter += 1
        for ddpg_agent in self.agents:
            soft_update(ddpg_agent.target_actor, ddpg_agent.actor, self.config.tau)
            soft_update(ddpg_agent.target_critic, ddpg_agent.critic, self.config.tau)
            
            
    def load_checkpoints(self):
        for i in range(len(self.agents)):
            self.agents[i].actor.load_state_dict( torch.load('checkpoints/actor'  + str(i) + 'checkpoint.pth'))
            self.agents[i].critic.load_state_dict(torch.load('checkpoints/critic' + str(i) + 'checkpoint.pth'))
