import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math
import sys
import torchtext
sys.path.append("..")
#from common import get_positional_encodings
import nltk

# Hyperparamètres

nb_decoders = 6
nb_heads = 8
vector_size = 512
vocab_size = 1000
head_size = vector_size//nb_heads
ffn_hidden_size = 100
seq_length = 15
attention_features_size = 5


class Transformer(nn.Module):
    "Whole transformer structure composed of stacked decoder blocks"
    def __init__(self, decoder):
        super(Transformer, self).__init__()
        self.decoder = decoder
        self.embedding = nn.Embedding(vocab_size, 4)
        self.finalfc = nn.Linear(vector_size, vocab_size)

    def forward(self, x):
        
        embedded = embedding(x)
        pos_encodings = torch.tensor(get_positional_encodings(seq_length, vector_size))
        x = embedded + pos_encodings
        
        for x in range(nb_decoders):
            x = self.decoder(x)
            
        x = F.softmax(self.finalfc(x))
        return x

    
class Decoder(nn.Module):
    "Decoder block applying multihead attention mechanism followed by a feedforward network"
    def __init__(self, multihead_attention, feedforward_network):
        super(Decoder, self).__init__()
        self.multihead_attention = multihead_attention
        self.feedforward_network = feedforward_network
        self.layernorm = nn.modules.normalization.LayerNorm

    def forward(self, x):
        #normalisation a la fin ou au debut ?
        mha = self.multihead_attention(self.layernorm(x))
        x += mha
        ffo = self.feedforward_network(self.layernorm(x))
        x += ffo
        return x

    
class FeedforwardNetwork(nn.Module):
    "Classic Feedforward Network with two layers"
    def __init__(self, vector_size, hidden_size):
        super(FeedforwardNetwork, self).__init__()
        self.hidden_size  = hidden_size
        self.fc1 = nn.Linear(vector_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, vector_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    

class MultiHeadAttention(nn.Module):
    "MultiHead Attention Block"
    def __init__(self, nb_heads, head_size, vector_size):
        super(MultiHeadAttention, self).__init__()
        self.nb_heads = nb_heads
        self.head_size = head_size
        self.vector_size = vector_size
        #bias = False ? (pour équivalence stricte avec une multiplication matricielle)
        self.w_q = nn.Linear(vector_size, vector_size)
        self.w_k = nn.Linear(vector_size, vector_size)
        self.w_v = nn.Linear(vector_size, vector_size)
        self.w_0 = nn.Linear(vector_size, vector_size)
        
    def mask_w(w):
        #Mask ?
        return w
    
    def reshape_w(self, w):
        #reshape a matrix (batch_size, nb_inputs, vector_size)
        #towards a matrix (batch_size, nb_heads, nb_inputs, head_size)
        w = w.reshape(-1, w.shape[1], self.nb_heads, self.head_size)
        w = w.transpose(-2,-3)
        return w
        
    def forward(self, x):
        
        # x size (batch_size, nb_inputs, vector_size)
        # self.w_q(x) size (batch_size, nb_inputs, vector_size)
        q = self.reshape_w(self.w_q(x))
        # q size (batch_size, nb_heads, nb_inputs, head_size)
        k = self.reshape_w(self.w_k(x))
        # k size (batch_size, nb_heads, nb_inputs, head_size)
        v = self.reshape_w(self.w_v(x))
        # v size (batch_size, nb_heads, nb_inputs, head_size)
        
        w = torch.matmul(q, k.transpose(-1,-2)) / math.sqrt(k.shape[-1])
        w = torch.softmax(w, dim=-1)
        # w size (batch_size, nb_heads, nb_inputs, nb_inputs)
        
        a = torch.matmul(w, v)
        # a size (batch_size, nb_heads, nb_inputs, head_size)
        
        a = a.transpose(-2,-3).reshape(-1, a.shape[-2], self.vector_size)
        # a size (batch_size, nb_inputs, vector_size) (concatenation of multihead matrices)
        
        a = self.w_0(a)
        # a size (batch_size, nb_inputs, vector_size)

        return a
    
