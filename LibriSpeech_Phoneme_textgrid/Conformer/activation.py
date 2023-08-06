import torch
import torch.nn as nn


class Swish(nn.Module):
    def __init__(self):
        super(Swish,self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        return x * self.sigmoid(x)

class GLU(nn.Module):
    def __init__(self, dim: int):
        super(GLU,self).__init__()
        self.dim = dim       
    
    def forward(self, inputs):
        outputs, gate = inputs.chunk(2, dim=self.dim)
        return outputs * gate.sigmoid()