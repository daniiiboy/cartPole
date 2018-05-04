# -*- coding: utf-8 -*-
"""
Created on Thu May  3 22:56:38 2018

@author: Daniyar AkhmedAngel

@sicription: common utility functions used for evaluating the performance of
             a model
"""
import torch
import numpy as np
from torch.autograd import Variable

def add_noise(state):
#        mu = 0
#        sigma = 0.3
#        for i in range(len(state)//2):
#            ind = np.random.randint(0, len(state))
#            state[ind] += np.random.normal(mu,sigma) # Add noise to randomly picked dimensions
#        norm_dim = np.random.normal(mu,sigma)        # Add dimension with normal noise
#        unif_dim = np.random.uniform(-sigma,sigma)   # Add dimension with uniform noise
        state = np.concatenate(([0], state, [0]))
        return state 

def evaluateFinalCartPole(env, policy, path = 'savedModel/simpleNet_Final', display = False, noise = False):
    policy.load_state_dict(torch.load(path))
    
    time_steps = []    # Array to store the length of each episode to monitor model learning
    for i_episode in range(100):
        policy.eval()
        state = env.reset() # Call this for new episode
        for t in range(200):
            if display:
               env.render()
            if noise:
                state = add_noise(state)   
            state = torch.from_numpy(state).float()
            pr = policy(Variable(state).type(torch.FloatTensor)).data.numpy()
            action = np.argmax(pr)
            state, reward, done, info = env.step(action)
    
            if done:
                time_steps.append(t+1)
                break
    if display:
        env.close() # !!! Need this to close the window
    return np.mean(time_steps)