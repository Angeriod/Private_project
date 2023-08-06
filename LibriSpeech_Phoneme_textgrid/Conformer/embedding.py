import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self,d_model=512, max_len=10000):
        super(PositionalEncoding,self).__init__()
        pe = torch.zeros(max_len, d_model, requires_grad=False)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) 
        # (max_len,1): 0 ~ (max_len-1) -> tensor of input sequence location information
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)) #  denominator of sin and cos
        # making 1 / (10000^(2i/d_model))
        # torch.exp(torch.arange(0, d_model, 2).float() -> make 0~d_model :even number, 
        pe[:,0::2] = torch.sin(position * div_term) # even index assigning from all the rows (max_len, d_model/2)
        pe[:,1::2] = torch.cos(position * div_term) # odds index assigning from all the rows
        pe = pe.unsqueeze(0) # Add batch dimension
        self.register_buffer('pe', pe) # will not be trained-> not for updating 

    def forward(self, length):
        return self.pe[:, :length]
