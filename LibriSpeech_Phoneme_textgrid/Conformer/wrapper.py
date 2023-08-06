import torch
import torch.nn as nn
import torch.nn.init as init

class ResidualConnectionModule(nn.Module):
    def __init__(self, module, module_factor = 1, input_factor = 1):
        super(ResidualConnectionModule,self).__init__()
        self.module = module
        self.module_factor = module_factor
        self.input_factor = input_factor

    def forward(self, inputs):
        return (self.module(inputs) * self.module_factor) + (inputs * self.input_factor)

class Linear(nn.Module): # originally linear is initalized with xavier_uniform
    def __init__(self,in_feature, out_feature, bias=True):
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_feature, out_feature, bias=bias)
        init.xavier_uniform_(self.linear.weight)
        if bias:
            init.zeros_(self.linear.bias)

    def forward(self,x):
        return self.linear(x)

class View(nn.Module):
    def __init__(self, shape: tuple, contiguous: bool = False):
        super(View,self).__init__()
        self.shape = shape
        self.contiguous = contiguous # reassigning memeory

    def forward(self, x):
        if self.contiguous:
            x = x.contiguous()
        
        return x.view(*self.shape)

class Transpose(nn.Module):
    def __init__(self, shape:tuple):
        super(Transpose, self).__init__()
        self.shape =shape
    
    def forward(self,x):
        return x.transpose(*self.shape)