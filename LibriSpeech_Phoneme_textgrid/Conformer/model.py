import torch
import torch.nn as nn

from .encoder import ConformerEncoder
from .wrapper import Linear

class Conformer(nn.Module):
    def __init__(self,num_classes, input_dim=80, encoder_dim=512, num_encoder_layers=17, num_attention_heads = 8, feed_forward_expansion_factor=4,
                 conv_expansion_factor =2, input_dropout_p =0.1, feed_forward_dropout_p = 0.1, attention_dropout_p =0.1, conv_dropout_p =0.1,
                 conv_kernel_size=31,half_step_residual=True):
        super(Conformer,self).__init__()
        self.encoder = ConformerEncoder(
            input_dim= input_dim,
            encoder_dim = encoder_dim,
            num_layers= num_encoder_layers,
            num_attention_heads = num_attention_heads,
            feed_forward_expansion_factor = feed_forward_expansion_factor,
            conv_expansion_factor = conv_expansion_factor,
            input_dropout_p = input_dropout_p,
            feed_forward_dropout_p = feed_forward_dropout_p,
            attention_dropout_p = attention_dropout_p,
            conv_dropout_p = conv_dropout_p,
            conv_kernel_size = conv_kernel_size,
            half_step_residual = half_step_residual
        )
     
        self.fc = Linear(encoder_dim, num_classes, bias=False)
        
    
    def count_parameter(self):
        return self.encoder.count_parameters()
    
    def update_dropout(self, dropout_p):
        self.encoder.update_dropout(dropout_p)

    def forward(self, inputs, input_lengths):
        encoder_outputs, encoder_output_lengths = self.encoder(inputs, input_lengths)

        outputs = self.fc(encoder_outputs)
        outputs = nn.functional.log_softmax(outputs, dim=-1)

        return outputs, encoder_output_lengths # predictions
        # inputs (batch,time,dim)
        # input_lengths (batch)
        # outputs (batch, out_channels, time)
        # output_lengths (batch)


 