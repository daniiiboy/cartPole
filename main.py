# -*- coding: utf-8 -*-
"""
Created on Thu May  3 00:19:43 2018

@author: Daniyar AkhmedAngel
"""
import gym
import model
import torch
from torch.autograd import Variable
import numpy as np


''' TIPS
Don't call env.render() to not show the gui with the cartpole
'''


def select_action(policy, state):
    """Samples an action from the policy at the state."""
    state = torch.from_numpy(state).long()
    pr = policy(Variable(state).type(torch.FloatTensor))
#    print(pr.data)
    m = torch.distributions.Categorical(pr)
    action = m.sample()
    log_prob = (m.log_prob(action))
    return action.data[0], log_prob

def finish_episode(saved_rewards, saved_logprobs, gamma=0.99):
    """Samples an action from the policy at the state."""
    policy_loss = []
    returns = compute_returns(saved_rewards, gamma)
#    print(returns)
    returns = torch.Tensor(returns)
    # subtract mean and std for faster training
    returns = (returns - returns.mean()) / (returns.std() +
                                            np.finfo(np.float32).eps)
    for log_prob, reward in zip(saved_logprobs, returns):
        policy_loss.append(-log_prob * reward)
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward(retain_graph=False)
    # note: retain_graph=True allows for multiple calls to .backward()
    # in a single step

def compute_returns(rewards, gamma=0.99):
    """
    Compute returns for each time step, given the rewards
      @param rewards: list of floats, where rewards[t] is the reward
                      obtained at time step t
      @param gamma: the discount factor
      @returns list of floats representing the episode's returns
          G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + ... 

    >>> compute_returns([0,0,0,1], 1.0)
    [1.0, 1.0, 1.0, 1.0]
    >>> compute_returns([0,0,0,1], 0.9)
    [0.7290000000000001, 0.81, 0.9, 1.0]
    >>> compute_returns([0,-0.5,5,0.5,-10], 0.9)
    [-2.5965000000000003, -2.8850000000000002, -2.6500000000000004, -8.5, -10.0]
    """
    com_return = [0]*len(rewards)
    com_return[-1] = rewards[-1]
    for i in reversed(range(len(rewards)-1)):
        com_return[i] = rewards[i] + gamma*com_return[i+1]
    return com_return

# Initialize the environment
env = gym.make('CartPole-v0')

# Initialize the model
policy = model.SimpleNet()

#Train the Model Using Policy Gradient
for i_episode in range(1000):
    state = env.reset() # Call this for new episode
    saved_rewards = []
    saved_logprobs = []
    for t in range(100):
#        env.render()
#        print(observation)
         
        action, logprob = select_action(policy, state)
        state, reward, done, info = env.step(action)
        
        reward = 1
        saved_logprobs.append(logprob)
        saved_rewards.append(reward)

        if done:
            print("Episode {} finished after {} timesteps".format(i_episode, t+1))
            break
#    print(saved_rewards)
    finish_episode(saved_rewards, saved_logprobs, 0.99)

#env.close() # !!! Need this to close the window
