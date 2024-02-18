import torch
import torch.nn as nn
from conformer import ConformerEncoder

class Conformer_L(nn.Module):
    def __init__(self,config):
        super(Conformer_L, self).__init__()
        self.model = ConformerEncoder(
            d_input=config.n_mels,
            d_model=config.d_model,
            num_layers=config.num_layers,
            conv_kernel_size=config.conv_kernel_size,
            feed_forward_residual_factor=config.feed_forward_residual_factor,
            feed_forward_expansion_factor=config.feed_forward_expansion_factor,
            dropout=config.drop_out
        )

    def forward(self, inputs,mask):
        output = self.model(inputs, mask)

        return output


class LSTMDecoder(nn.Module):
    def __init__(self,config):
        super(LSTMDecoder,self).__init__()

        self.lstm = nn.LSTM(input_size=config.d_model, hidden_size=config.d_decoder,num_layers=1,batch_first=True)
        self.linear = nn.Linear(config.d_decoder,config.num_classes)

    def forward(self, x):
        outputs, _ = self.lstm(x)
        logits = self.linear(outputs)

        return logits

class GreedyCharacterDecoder(nn.Module):
    def __init__(self):
        super(GreedyCharacterDecoder, self).__init__()

    def forward(self,x):
        indices = torch.argmax(x, dim=-1)
        indices = torch.unique_consecutive(indices,dim=-1)

        return indices.tolist()

'''
batch_size, sequence_length, dim = 3, 12345, 80

cuda = torch.cuda.is_available()  
device = torch.device('cuda' if cuda else 'cpu')

criterion = nn.CTCLoss().to(device)

inputs = torch.rand(batch_size, sequence_length, dim)
input_lengths = torch.LongTensor([12345, 12300, 12000])
targets = torch.LongTensor([[1, 3, 3, 3, 3, 3, 4, 5, 6, 2],
                            [1, 3, 3, 3, 3, 3, 4, 5, 2, 0],
                            [1, 3, 3, 3, 3, 3, 4, 2, 0, 0]])
target_lengths = torch.LongTensor([9, 8, 7])

model = Conformer(num_classes=10, 
                  input_dim=dim, 
                  encoder_dim=32, 
                  num_encoder_layers=3)

# Forward propagate
outputs, output_lengths = model(inputs, input_lengths)

print(outputs.shape)
print(output_lengths.shape)
print(targets.shape)
print(target_lengths.shape)
# Calculate CTC Loss
loss = criterion(outputs.transpose(0, 1), targets, output_lengths, target_lengths)
'''
