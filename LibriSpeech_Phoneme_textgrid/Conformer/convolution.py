import torch
import torch.nn as nn
from .activation import GLU, Swish
from .wrapper import Transpose

class PointwiseConv1d(nn.Module): # (channels use different weights) means bias are needed for each channels -> bias=True
    def __init__(self, in_channels, out_channels,kernel_size,stride=1,padding=0,bias=True):
        super(PointwiseConv1d,self).__init__()
        self.conv=nn.Conv1d(in_channels,out_channels,kernel_size=1,groups=in_channels,stride=stride,padding=padding,bias=bias)
        #groups
    def forward(self,x):
        return self.conv(x)
    # input (batch, in_channels, time), output (batch, out_channels, time)

class DepthwiseConv1d(nn.Module):# bias vary between kernels-> bias=False 
    def __init__(self, in_channels, out_channels,kernel_size,stride=1,padding=0,bias=False):
        super(DepthwiseConv1d,self).__init__()
        self.conv =nn.Conv1d(in_channels,out_channels,kernel_size=kernel_size,stride=1,padding=padding,bias=bias)

    def forward(self,x):
        return self.conv(x)
    # input (batch, in_channels, time), output (batch, out_channels, time)

class ConformerConvmodule(nn.Module):
    def __init__(self, in_channels, kernel_size=31 , expansion_factor=2, dropout_p=0.1):# kernel size should be odd for "SAME" padding
        super(ConformerConvmodule,self).__init__()
        #expansion factor 2 is in paper
        self.sequential = nn.Sequential(
            nn.LayerNorm(in_channels),
            Transpose(shape=(1, 2)),
            #PointwiseConv1d(in_channels, in_channels * expansion_factor, stride=1, padding=0, bias=True), 
            PointwiseConv1d(in_channels,in_channels*expansion_factor,kernel_size=1,stride=1,padding=0,bias=True),
            # After GLU, in_channels will be divied with 2 
            # input:(batch, in_channels, time), output:(batch, out_channels, time)
            GLU(dim = 1), 
            DepthwiseConv1d(in_channels, in_channels,kernel_size=kernel_size,stride=1,padding=(kernel_size-1)//2,bias=False),
            #DepwiseConv1d -> kenerls have to be applied with channels sperately 
            nn.BatchNorm1d(in_channels),
            Swish(),
            PointwiseConv1d(in_channels,in_channels,kernel_size=1,stride=1,padding=0,bias=True),# pointwiseConv1d
            nn.Dropout(p=dropout_p),
        )
        
    def forward(self,x):
        return self.sequential(x).transpose(1,2)
        # input: (batch, time, dim) # output:(batch, time, dim)

class Conv2dSubsampling(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(Conv2dSubsampling,self).__init__()
        self.sequential = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3,stride=2),
            nn.ReLU(),
        )
    def forward(self, inputs, input_lengths):
        outputs = self.sequential(inputs.unsqueeze(1))# input of conv2d must be 4 dimensions.-> by "unsqueeze" make channel dimension
        batch_size, channels, subsampled_lenghts, subsampled_dim = outputs.size()

        outputs = outputs.permute(0,2,1,3)
        outputs = outputs.contiguous().view(batch_size, subsampled_lenghts, channels * subsampled_dim)# reduction

        output_lengths = input_lengths >> 1 # 1/4-> 1/2 because of the short waveform length  
        
        output_lengths -= 1
        
        return outputs, output_lengths
