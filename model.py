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
        self.linear1 = nn.Linear(dim_in, 32)
        self.linear2 = nn.Linear(32, dim_out)

    def forward(self, inputs):
        l1 = F.relu(self.linear1(inputs))
        l2 = self.linear2(l1)
        log_probs = F.softmax(l2, dim=0)
        return log_probs
    
    def saveWeights(self, fileName = 'simpleNet_default'):
        torch.save(self.state_dict(), 'savedModel/'+fileName)

class NGramLanguageModeler(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NGramLanguageModeler, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)
        
    def word2vec(self,word_ind):
        lookup_tensor = torch.LongTensor([word_ind])
        looked_up = self.embeddings(autograd.Variable(lookup_tensor))
        return looked_up.data.numpy()
        
    def forward(self, inputs):
        embeds = self.embeddings(inputs)
        embeds = embeds.view((inputs.shape[0], -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs

def saveEmbeddings(size,X,Y,use_gpu=False,epoch_num=100, alpha=0.001, bs = 1000):
    # X: [N x CONTEXT_SIZE] - Contains the indeces of the context words
    # Y: [N x 1]  - Contains the target words one hot key index
    # size: N
    loss_function = nn.NLLLoss()
    model = NGramLanguageModeler(size, 5,5)
    optimizer = optim.Adam(model.parameters(), lr=alpha)
    
    for seed in [44]: #132#44#12#5
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)  
        
        # Check if GPU is available
        if use_gpu:
            model.cuda()
            floatTensor = torch.cuda.FloatTensor
            longTensor = torch.cuda.LongTensor
        else:
            floatTensor = torch.FloatTensor
            longTensor = torch.LongTensor

    for epoch in range(epoch_num):
        print ("EPOCH", epoch)
        itter = X.shape[0]/bs
        shuffled = np.arange(int(itter))
        np.random.shuffle(shuffled)
        for i in shuffled:
            X_n = X[int(i*bs):int((i+1)*bs)]
            x = autograd.Variable(torch.from_numpy(X_n), requires_grad=False).type(longTensor)
            Y_n = Y[int(i*bs):int((i+1)*bs)]
            y = autograd.Variable(torch.from_numpy(Y_n), requires_grad=False).type(longTensor)

            model.zero_grad()
            log_probs = model(x)
            loss = loss_function(log_probs, y.squeeze())
            loss.backward()
            optimizer.step()
    
        if epoch%10==0:
            print('LOSS:',loss.data[0])
            torch.save(model.state_dict(),'models/NGram/ngram_model3')
        
            
