import torch
import torch.nn as nn
from Conformer import Conformer


class Conformer_L(nn.Module):
    def __init__(self,config):
        super(Conformer_L, self).__init__()
        self.model= Conformer(num_classes=config.num_classes,
                input_dim=config.n_mels)

    def forward(self, inputs, input_lengths):
        outputs, output_legnths=self.model(inputs, input_lengths)
        return outputs, output_legnths

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
