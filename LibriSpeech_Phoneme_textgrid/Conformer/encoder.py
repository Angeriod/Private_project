import torch
import torch.nn as nn
from .feed_forward import FeedForwardModule
from .attention import MultiHeadedSelfAttentionModule
from .convolution import (ConformerConvmodule, Conv2dSubsampling)
from .wrapper import (ResidualConnectionModule, Linear)

class ConformerBlock(nn.Module): # Conformer L base
    def __init__(self,encoder_dim = 512,num_attention_heads = 8,feed_forward_expansion_factor = 4,conv_expansion_factor = 2,
            feed_forward_dropout_p = 0.1,attention_dropout_p = 0.1,conv_dropout_p = 0.1,conv_kernel_size = 31,
            half_step_residual = True):
        super(ConformerBlock,self).__init__()
        if half_step_residual:
            self.feed_forward_residual_factor = 0.5
        else:
            self.feed_forward_residual_factor = 1

        self.sequential = nn.Sequential(
            ResidualConnectionModule(
                module=FeedForwardModule(
                    encoder_dim=encoder_dim,
                    expansion_factor=feed_forward_expansion_factor,
                    dropout_p=feed_forward_dropout_p,
                ),
                module_factor=self.feed_forward_residual_factor,
            ),
            ResidualConnectionModule(
                module=MultiHeadedSelfAttentionModule(
                    d_model=encoder_dim,
                    num_heads=num_attention_heads,
                    dropout_p=attention_dropout_p,
                ),
            ),
            ResidualConnectionModule(
                module=ConformerConvmodule(
                    in_channels=encoder_dim,
                    kernel_size=conv_kernel_size,
                    expansion_factor=conv_expansion_factor,
                    dropout_p=conv_dropout_p,
                ),
            ),
            ResidualConnectionModule(
                module=FeedForwardModule(
                    encoder_dim=encoder_dim,
                    expansion_factor=feed_forward_expansion_factor,
                    dropout_p=feed_forward_dropout_p,
                ),
                module_factor=self.feed_forward_residual_factor,
            ),
            nn.LayerNorm(encoder_dim),
        )

        
    def forward(self, inputs, mask=None):
        return self.sequential(inputs)
        
class ConformerEncoder(nn.Module):
    def __init__(self, input_dim = 80, encoder_dim = 512, num_layers = 17, num_attention_heads = 8, feed_forward_expansion_factor= 4,
                conv_expansion_factor = 2, input_dropout_p = 0.1, feed_forward_dropout_p =0.1, attention_dropout_p =0.1, conv_dropout_p =0.1,
                conv_kernel_size = 31, half_step_residual = True ):
        super(ConformerEncoder,self).__init__()
        self.conv_subsample = Conv2dSubsampling(in_channels = 1, out_channels = encoder_dim)
        self.input_projection = nn.Sequential(
            Linear(encoder_dim * ((input_dim - 1) // 2), encoder_dim), # 1/2 -> -1 1/3 -> -2
            # matching subsampling length with encoder block dimension
            nn.Dropout(p=input_dropout_p)
        )

        self.layers = nn.ModuleList([ConformerBlock(
            encoder_dim = encoder_dim,
            num_attention_heads = num_attention_heads,
            feed_forward_expansion_factor = feed_forward_expansion_factor,
            conv_expansion_factor = conv_expansion_factor,
            feed_forward_dropout_p = feed_forward_dropout_p,
            attention_dropout_p = attention_dropout_p,
            conv_dropout_p = conv_dropout_p,
            conv_kernel_size = conv_kernel_size,
            half_step_residual = half_step_residual
        ) for _ in range(num_layers)])

    def count_parameters(self):
        return sum([p.numel() for p in self.parameters()])
    
    def update_dropout(self, dropout_p):
        for name, child in self.named_children():
            if isinstance(child, nn.Dropout):
                child.p = dropout_p
    
    def forward(self, inputs, input_lengths,mask=None):

        outputs, output_lengths = self.conv_subsample(inputs, input_lengths)
        outputs = self.input_projection(outputs)

        for layer in self.layers:
            outputs = layer(outputs)
 
        return outputs, output_lengths 
    
        # inputs (batch,time,dim)
        # input_lengths (batch)
        #outputs = (batch, out_channels, time)
        #outputs_lengths = (batch)