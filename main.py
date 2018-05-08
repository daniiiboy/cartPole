# -*- coding: utf-8 -*-
"""
Created on Thu May  3 00:19:43 2018

@author: Daniyar AkhmedAngel
"""    
from policyGradient import PolicyGradientGym
import gym
import model
import evaluation_util as util
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    # Initialize PolicyGradient Training class
    policyCartPole = PolicyGradientGym()
    
     # Initialize the environment
    env = gym.make('CartPole-v0')
    
    # Initialize the model
    policy = model.SimpleNet(dim_in = 6) # Set dim=4 if no noise or dim=6 if noise is added
    
     #Train the Model Using Policy Gradient    
#    policyCartPole.train(env, policy, noise=True, printUpdate = True)
     
     # Evaluute the learned policy
#    print('Final Evluation: Average Reward for 100 Episodes')
#    print(util.evaluateFinalCartPole(env, policy, noise=True))
#    util.learningCurve(env, policy, noise=True, save_path='performance/simpleNet_performance')
#    util.plotLearningCurve('performance/simpleNet_performance')
    
    
    
    # HYPERPARAMETER TUNING
    # buffer_size
    # alpha
    # discount factor
    buff = 1
    alpha = 0.1
    d_factor = 1.0
    bs = []
    als =[]
    ds = []
    perfs = []
    
    for i in range(20):
        buff = 1
        print('TIME {}'.format(i))
        for b in range(5):
            print('Buffer Size {}'.format(buff))
            alpha = 0.1
            for a in range(10):
                print('   Learning Rate {:.6f}'.format(alpha))
                d_factor = 1.0
                for d in range(10):
                    print('      Discount Factor {:.2f}'.format(d_factor))
                    policy = model.SimpleNet(dim_in = 6) 
                    policyCartPole.train(env, policy, noise=True, printUpdate = False, n_episodes=300, save_model= 50)
                    
                    perfs.append(util.evaluateFinalCartPole(env, policy, noise=True))
                    print('                          {:3.2f}'.format(perfs[-1]))
                    
                    save_path = ('performance/simpleNet_ave_b{}_a{:.6f}_df{:.3f}'.format(buff,alpha,d_factor))
                    util.learningCurve(env, policy, noise=True, save_path=save_path, save_model_n=50, n_episodes=300)
                    
                    bs.append(buff)
                    als.append(alpha)
                    ds.append(d_factor)
                    
                    d_factor -=0.005
                alpha /= 2
            buff *=2
            
        final = np.vstack((bs,als,ds,perfs))
        path_name = 'performanceAve/hyperParamTune_{}ave_episode300_saveModel50'.format(i)
        np.save(path_name, final)
    
     