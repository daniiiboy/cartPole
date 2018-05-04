# -*- coding: utf-8 -*-
"""
Created on Thu May  3 00:55:28 2018

@author: Daniyar AkhmedAngel
"""

import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

torch.manual_seed(1)

class SimpleNet(nn.Module):
    '''
        Simple Neural Net
    '''
    def __init__(self, dim_in=4, dim_out=2):
        super(SimpleNet, self).__init__()
        self.linear1 = nn.Linear(dim_in, 128)
        self.linear2 = nn.Linear(128, dim_out)

    def forward(self, inputs):
        l1 = F.relu(self.linear1(inputs))
        l2 = self.linear2(l1)
        log_probs = F.softmax(l2, dim=0)
        return log_probs
