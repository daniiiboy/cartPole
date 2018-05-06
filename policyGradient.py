# -*- coding: utf-8 -*-
"""
Created on Thu May  3 23:15:06 2018

@author: Daniyar AkhmedAngel
"""

import torch
from torch.autograd import Variable
import numpy as np

''' TIPS
To train quickly
don't call env.render() to not show the gui with the cartpole
'''
class PolicyGradientGym(object):
    def __init__(self, tiral_num = 'CartPole'):
        self.trial = tiral_num
        
    def add_noise(self, state):
        mu = 0
        sigma = 0.3
        for i in range(len(state)//2):
            ind = np.random.randint(0, len(state))
            state[ind] += np.random.normal(mu,sigma) # Add noise to randomly picked dimensions
        norm_dim = np.random.normal(mu,sigma)        # Add dimension with normal noise
        unif_dim = np.random.uniform(-sigma,sigma)   # Add dimension with uniform noise
        state = np.concatenate(([norm_dim], state, [unif_dim]))
        return state                
        
    def select_action(self,policy, state, noise = False):
        """Samples an action from the policy at the state."""
        if noise:
            state = self.add_noise(state)
        state = torch.from_numpy(state).float()
        pr = policy(Variable(state).type(torch.FloatTensor))
        m = torch.distributions.Categorical(pr)
        action = m.sample()
        log_prob = (m.log_prob(action))
        return action.data[0], log_prob
    
    def finish_episode(self,saved_rewards, saved_logprobs, gamma=1):
        """Samples an action from the policy at the state."""
        policy_loss = 0
        returns = self.compute_returns(saved_rewards, gamma)
        
        returns = torch.Tensor(returns)
        # subtract mean and std for faster training
    #    returns = (returns - returns.mean()) / (returns.std()*2 +
    #                                            np.finfo(np.float32).eps)
        for log_prob, reward in zip(saved_logprobs, returns):
            policy_loss += -log_prob * reward
#        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward(retain_graph=False)
        # note: retain_graph=True allows for multiple calls to .backward()
        # in a single step
    
    def compute_returns(self,rewards, gamma=1):
        """
        Compute returns for each time step, given the rewards
          @param rewards: list of floats, where rewards[t] is the reward
                          obtained at time step t
          @param gamma: the discount factor
          @returns list of floats representing the episode's returns
              G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + ... 
        """
        com_return = [0]*len(rewards)
        com_return[-1] = rewards[-1]
        for i in reversed(range(len(rewards)-1)):
            com_return[i] = rewards[i] + gamma*com_return[i+1]
        return com_return
    
    def saveWeights(self, model, fileName = 'simpleNet_default'):
        torch.save(model.state_dict(), 'savedModel/'+fileName)
    
    def train(self, env, policy, 
              trial_id = '', n_episodes = 3000, episode_len = 200, 
              save_model = 100, buffer_size = 5, alpha = 0.002, 
              discount_fact = 0.99, 
              display = False, noise = False, printUpdate= False):
        ''' Example @arguments
            env - gym environment that we want to train on
            policy - policy we are triaining
            trial_id = For naming the model
            n_episodes = 3000  # Number of trining episodes
            episode_len = 200  # Length of each episode
            save_model = 100   # Number of episodes between saving model weights
            buffer_size = 5    # Pseudo buffer size, number of episodes played before updating gradients
            alpha = 0.002       # Learning Rate
            discount_fact = 0.99 # Discount factor for rewards 
            display - wheather we want to display GUI while training
            
            noise = T/F   # Add noise to random half of the states, noise is sampled from normal dist.
                          # Add addional two states: one sampled from normal distribution
                                                     second sampled from uniform
                          # To Do Intriguce Lag
        '''
        time_steps = []    # Array to store the length of each episode to monitor model learning
        for i_episode in range(n_episodes):
            optimizer = torch.optim.Adam(policy.parameters(), lr=alpha)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=save_model, gamma=0.95)
            state = env.reset() # Call this for new episode
            saved_rewards = []
            saved_logprobs = []
            for t in range(episode_len):
                if display:
                   env.render()
                action, logprob = self.select_action(policy, state, noise=noise)
                state, reward, done, info = env.step(action)
                
                reward = 1
                saved_logprobs.append(logprob)
                saved_rewards.append(reward)
        
                if done:
                    saved_rewards[-1] = 0
                    time_steps.append(t+1)
                    saved_rewards[-1] = 1
                    break
                
            if (i_episode) % save_model == 0:
                if printUpdate:
                    print("Episode {} finished after {:3.2f} average timesteps".format(i_episode, np.mean(time_steps)))
                time_steps = [] # Reset time_steps
                name = str(trial_id)+ 'simpleNet_' + str(i_episode)
                self.saveWeights(policy, name)
            
            # Pseudo Replay Buffer - Decreases Variance in Rewards - Helps to Learn quicker
            if i_episode % buffer_size == 0: # batch_size
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
            self.finish_episode(saved_rewards, saved_logprobs, discount_fact)
            
        if display:
            env.close() # !!! Need this to close the window
            
        name = str(trial_id)+ 'simpleNet_' + str(n_episodes)
        self.saveWeights(policy, name)
        if printUpdate:
            print("Episode {} finished after {:3.2f} average timesteps".format(n_episodes, np.mean(time_steps)))
        
        name = str(trial_id)+ 'simpleNet_Final'
        self.saveWeights(policy, name)