# -*- coding: utf-8 -*-
"""
Created on Thu May  3 00:19:43 2018

@author: Daniyar AkhmedAngel
"""    
from policyGradient import PolicyGradientGym
import gym
import model
from evaluation_util import evaluateFinalCartPole

if __name__ == "__main__":
    # Initialize PolicyGradient Training class
    policyCartPole = PolicyGradientGym()
     # Initialize the environment
    env = gym.make('CartPole-v0')
    # Initialize the model
    policy = model.SimpleNet(dim_in = 4)
    
     #Train the Model Using Policy Gradient    
    policyCartPole.train(env, policy, noise=False)
     
     # Evaluute the final policy on 200 episodes
    print('Final Evluation: Average Reward for 100 Episodes')
    print(evaluateFinalCartPole(env, policy, noise=False))
     
     