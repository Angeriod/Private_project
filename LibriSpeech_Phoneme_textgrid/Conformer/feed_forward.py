import torch
import torch.nn as nn

from .activation import Swish
from .wrapper import Linear

class FeedForwardModule(nn.Module):
    def __init__(self,encoder_dim = 512, expansion_factor = 4, dropout_p = 0.1): # expansion_fator = 4 is in paper
        super(FeedForwardModule, self).__init__()
        self.sequential = nn.Sequential(
            nn.LayerNorm(encoder_dim),
            Linear(encoder_dim, encoder_dim * expansion_factor, bias=True),
            Swish(),
            nn.Dropout(p=dropout_p),
            Linear(encoder_dim * expansion_factor, encoder_dim, bias=True),
            nn.Dropout(p=dropout_p)
        )

    def forward(self, inputs):
        return self.sequential(inputs)
