import os
import random

import torch
import torch.nn as nn
import torchaudio

import math
import torch
from torch import nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchaudio.datasets import LIBRISPEECH

import gc
from torch.cuda.amp import autocast, GradScaler
from torchmetrics.text.wer import WordErrorRate

def train(encoder, decoder, char_decoder, optimizer, scheduler, criterion, grad_scaler, train_loader, gpu=True):
    ''' Run a single training epoch '''

    wer = WordErrorRate()
    error_rate = AvgMeter()
    avg_loss = AvgMeter()
    text_transform = TextTransform()

    encoder.train()
    decoder.train()
    for i, batch in enumerate(train_loader):
        scheduler.step()
        gc.collect()
        spectrograms, labels, input_lengths, label_lengths, references, mask = batch 
      
        # Move to GPU
        if gpu:
            spectrograms = spectrograms.cuda()
            labels = labels.cuda()
            input_lengths = torch.tensor(input_lengths).cuda()
            label_lengths = torch.tensor(label_lengths).cuda()
            mask = mask.cuda()
      
        # Update models
        with autocast(enabled=False): #16비트 부동소수점로 수행
            outputs = encoder(spectrograms, mask)
            outputs = decoder(outputs)
            loss = criterion(F.log_softmax(outputs, dim=-1).transpose(0, 1), labels, input_lengths, label_lengths)
        grad_scaler.scale(loss).backward() # 기울기의 값들을 일시적으로 더 큰 값으로 조정(scaling up)하여 낮은 정밀도에서의 연산으로 인한 데이터 손실을 방지합니다.
        if (i + 1) % 1 == 0:
            grad_scaler.step(optimizer)
            grad_scaler.update()
            optimizer.zero_grad()
        avg_loss.update(loss.detach().item())

        # Predict words, compute WER
        inds = char_decoder(outputs.detach())
        predictions = []
        for sample in inds:
            predictions.append(text_transform.int_to_text(sample))
        error_rate.update(wer(predictions, references) * 100)

        # Print metrics and predictions 
        if (i + 1) % 100 == 0:
            print(f'Step {i + 1} - Avg WER: {error_rate.avg}%, Avg Loss: {avg_loss.avg}')   
        del spectrograms, labels, input_lengths, label_lengths, references, outputs, inds, predictions
    return error_rate.avg, avg_loss.avg

def validate(encoder, decoder, char_decoder, criterion, test_loader, gpu=True):
    ''' Evaluate model on test dataset. '''

    avg_loss = AvgMeter()
    error_rate = AvgMeter()
    wer = WordErrorRate()
    text_transform = TextTransform()

    encoder.eval()
    decoder.eval()
    for i, batch in enumerate(test_loader):
        gc.collect()
        spectrograms, labels, input_lengths, label_lengths, references, mask = batch 
      
        # Move to GPU
        if gpu:
            spectrograms = spectrograms.cuda()
            labels = labels.cuda()
            input_lengths = torch.tensor(input_lengths).cuda()
            label_lengths = torch.tensor(label_lengths).cuda()
            mask = mask.cuda()

        with torch.no_grad():
            with autocast(enabled=False):
                outputs = encoder(spectrograms, mask)
                outputs = decoder(outputs)
                loss = criterion(F.log_softmax(outputs, dim=-1).transpose(0, 1), labels, input_lengths, label_lengths)
            avg_loss.update(loss.item())

            inds = char_decoder(outputs.detach())
            predictions = []
            for sample in inds:
                predictions.append(text_transform.int_to_text(sample))
            error_rate.update(wer(predictions, references) * 100)
    return error_rate.avg, avg_loss.avg

class AvgMeter(object):
  '''
    Keep running average for a metric
  '''
  def __init__(self):
    self.reset()

  def reset(self):
    self.avg = None
    self.sum = None
    self.cnt = 0

  def update(self, val, n=1):
    if not self.sum:
      self.sum = val * n 
    else:
      self.sum += val * n
    self.cnt += n
    self.avg = self.sum / self.cnt

def add_model_noise(model, std=0.0001, gpu=True):
  with torch.no_grad():
    for param in model.parameters():
        if gpu:
          param.add_(torch.randn(param.size()).cuda() * std)
        else:
          param.add_(torch.randn(param.size()).cuda() * std)

class PositionalEncoder(nn.Module):
  
  def __init__(self, d_model, max_len=10000):
      super(PositionalEncoder, self).__init__()
      self.d_model = d_model
      encodings = torch.zeros(max_len, d_model)
      pos = torch.arange(0, max_len, dtype=torch.float)
      inv_freq = 1 / (10000 ** (torch.arange(0.0, d_model, 2.0) / d_model))
      encodings[:, 0::2] = torch.sin(pos[:, None] * inv_freq)
      encodings[:, 1::2] = torch.cos(pos[:, None] * inv_freq)
      self.register_buffer('encodings', encodings)
    
  def forward(self, len):
      return self.encodings[:len, :]

class RelativeMultiHeadAttention(nn.Module):
  
  def __init__(self, d_model=144, num_heads=4, dropout=0.1, positional_encoder=PositionalEncoder(144)):
      super(RelativeMultiHeadAttention, self).__init__()

      #dimensions
      assert d_model % num_heads == 0
      self.d_model = d_model
      self.d_head = d_model // num_heads
      self.num_heads = num_heads

      # Linear projection weights
      self.W_q = nn.Linear(d_model, d_model)
      self.W_k = nn.Linear(d_model, d_model)
      self.W_v = nn.Linear(d_model, d_model)
      self.W_pos = nn.Linear(d_model, d_model, bias=False)
      self.W_out = nn.Linear(d_model, d_model)

      # Trainable bias parameters
      self.u = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))
      self.v = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))
      torch.nn.init.xavier_uniform_(self.u)
      torch.nn.init.xavier_uniform_(self.v)

      # etc
      self.layer_norm = nn.LayerNorm(d_model, eps=6.1e-5)
      self.positional_encoder = positional_encoder
      self.dropout = nn.Dropout(dropout)

  def forward(self, x, mask=None):
      batch_size, seq_length, _ = x.size()

      #layer norm and pos embeddings
      x = self.layer_norm(x)
      pos_emb = self.positional_encoder(seq_length)
      pos_emb = pos_emb.repeat(batch_size, 1, 1)

      #Linear projections, split into heads
      q = self.W_q(x).view(batch_size, seq_length, self.num_heads, self.d_head)
      k = self.W_k(x).view(batch_size, seq_length, self.num_heads, self.d_head).permute(0, 2, 3, 1) # (batch_size, num_heads, d_head, time)
      v = self.W_v(x).view(batch_size, seq_length, self.num_heads, self.d_head).permute(0, 2, 3, 1) # (batch_size, num_heads, d_head, time)
      pos_emb = self.W_pos(pos_emb).view(batch_size, -1, self.num_heads, self.d_head).permute(0, 2, 3, 1) # (batch_size, num_heads, d_head, time)

      #Compute attention scores with relative position embeddings
      AC = torch.matmul((q + self.u).transpose(1, 2), k)
      BD = torch.matmul((q + self.v).transpose(1, 2), pos_emb)
      BD = self.rel_shift(BD)
      attn = (AC + BD) / math.sqrt(self.d_model)

      #Mask before softmax with large negative number
      if mask is not None:
          mask = mask.unsqueeze(1)
          mask_value = -1e+30 if attn.dtype == torch.float32 else -1e+4
          attn.masked_fill_(mask, mask_value)

      #Softmax
      attn = F.softmax(attn, -1)

      #Construct outputs from values
      output = torch.matmul(attn, v.transpose(2, 3)).transpose(1, 2) # (batch_size, time, num_heads, d_head)
      output = output.contiguous().view(batch_size, -1, self.d_model) # (batch_size, time, d_model)

      #Output projections and dropout
      output = self.W_out(output)
      return self.dropout(output)


  def rel_shift(self, emb):
      
      batch_size, num_heads, seq_length1, seq_length2 = emb.size()
      zeros = emb.new_zeros(batch_size, num_heads, seq_length1, 1)
      padded_emb = torch.cat([zeros, emb], dim=-1)
      padded_emb = padded_emb.view(batch_size, num_heads, seq_length2 + 1, seq_length1)
      shifted_emb = padded_emb[:, :, 1:].view_as(emb)
      return shifted_emb


class ConvBlock(nn.Module):
    
    def __init__(self, d_model=144, kernel_size=31, dropout=0.1):
        super(ConvBlock, self).__init__()
        self.layer_norm = nn.LayerNorm(d_model, eps=6.1e-5)
        kernel_size=31
        self.module = nn.Sequential(
          nn.Conv1d(in_channels=d_model, out_channels=d_model * 2, kernel_size=1), # first pointwise with 2x expansion
          nn.GLU(dim=1),
          nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=kernel_size, padding='same', groups=d_model), # depthwise
          nn.BatchNorm1d(d_model, eps=6.1e-5),
          nn.SiLU(), # swish activation
          nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=1), # second pointwise
          nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.layer_norm(x)
        x = x.transpose(1, 2) # (batch_size, d_model, seq_len)
        x = self.module(x)
        return x.transpose(1, 2)

class FeedForwardBlock(nn.Module):
    
    def __init__(self, d_model=144, expansion=4, dropout=0.1):
        super(FeedForwardBlock, self).__init__()
        self.module = nn.Sequential(
          nn.LayerNorm(d_model, eps=6.1e-5),
          nn.Linear(d_model, d_model * expansion), # expand to d_model * expansion
          nn.SiLU(), # swish activation
          nn.Dropout(dropout),
          nn.Linear(d_model * expansion, d_model), # project back to d_model
          nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.module(x)

class Conv2dSubsampling(nn.Module):
    
    def __init__(self, d_model=144):
        super(Conv2dSubsampling, self).__init__()
        self.module = nn.Sequential(
          nn.Conv2d(in_channels=1, out_channels=d_model, kernel_size=3, stride=2),
          nn.ReLU(),
          nn.Conv2d(in_channels=d_model, out_channels=d_model, kernel_size=3, stride=2),
          nn.ReLU(),
        )

    def forward(self, x):
        output = self.module(x.unsqueeze(1)) # (batch_size, 1, time, d_input)
        batch_size, d_model, subsampled_time, subsampled_freq = output.size()
        output = output.permute(0, 2, 1, 3)
        output = output.contiguous().view(batch_size, subsampled_time, d_model * subsampled_freq)
        return output

class ConformerBlock(nn.Module):
    
    def __init__(
            self,
            d_model=144,
            conv_kernel_size=31,
            feed_forward_residual_factor=.5,
            feed_forward_expansion_factor=4,
            num_heads=4,
            positional_encoder=PositionalEncoder(144),
            dropout=0.1,
    ):
      super(ConformerBlock, self).__init__()
      self.residual_factor = feed_forward_residual_factor
      self.ff1 = FeedForwardBlock(d_model, feed_forward_expansion_factor, dropout)
      self.attention = RelativeMultiHeadAttention(d_model, num_heads, dropout, positional_encoder)
      self.conv_block = ConvBlock(d_model, conv_kernel_size, dropout)
      self.ff2 = FeedForwardBlock(d_model, feed_forward_expansion_factor, dropout)
      self.layer_norm = nn.LayerNorm(d_model, eps=6.1e-5)

    def forward(self, x, mask=None):
        x = x + (self.residual_factor * self.ff1(x))
        x = x + self.attention(x, mask=mask)
        x = x + self.conv_block(x)
        x = x + (self.residual_factor * self.ff2(x))
        return self.layer_norm(x)


class ConformerEncoder(nn.Module):
    '''
      Conformer Encoder Module. 

      Parameters:
        d_input (int): Dimension of the input
        d_model (int): Dimension of the model
        num_layers (int): Number of conformer blocks to use in the encoder
        conv_kernel_size (int): Size of kernel to use for depthwise convolution
        feed_forward_residual_factor (float): output_weight for feed-forward residual connections
        feed_forward_expansion_factor (int): Expansion factor for feed-forward block
        num_heads (int): Number of heads to use for multi-head attention
        dropout (float): Dropout probability
      
      Inputs:
        x (Tensor): input spectrogram of dimension (batch_size, time, d_input)
        mask (Tensor): (batch_size, time, time) Optional mask to zero out attention score at certain indices
      
      Outputs:
        Tensor (batch_size, time, d_model): Output tensor from the conformer encoder

    
    '''
    def __init__(
            self,
            d_input=80,
            d_model=144,
            num_layers=16,
            conv_kernel_size=31, 
            feed_forward_residual_factor=.5,
            feed_forward_expansion_factor=4,
            num_heads=4,
            dropout=.1,
    ):
      super(ConformerEncoder, self).__init__()
      self.conv_subsample = Conv2dSubsampling(d_model=d_model)
      self.linear_proj = nn.Linear(d_model * (((d_input - 1) // 2 - 1) // 2), d_model) # project subsamples to d_model
      self.dropout = nn.Dropout(p=dropout)
      
      # define global positional encoder to limit model parameters
      positional_encoder = PositionalEncoder(d_model) 
      self.layers = nn.ModuleList([ConformerBlock(
              d_model=d_model,
              conv_kernel_size=conv_kernel_size, 
              feed_forward_residual_factor=feed_forward_residual_factor,
              feed_forward_expansion_factor=feed_forward_expansion_factor,
              num_heads=num_heads,
              positional_encoder=positional_encoder,
              dropout=dropout,
          ) for _ in range(num_layers)])

    def forward(self, x, mask=None):
        x = self.conv_subsample(x)
        if mask is not None:
            mask = mask[:, :-2:2, :-2:2] #account for subsampling
            mask = mask[:, :-2:2, :-2:2] #account for subsampling
            assert mask.shape[1] == x.shape[1], f'{mask.shape} {x.shape}'
        
        x = self.linear_proj(x)
        x = self.dropout(x)
        
        for layer in self.layers:
            x = layer(x, mask=mask)
      
        return x


class LSTMDecoder(nn.Module):
    '''
      LSTM Decoder

      Parameters:
        d_encoder (int): Output dimension of the encoder
        d_decoder (int): Hidden dimension of the decoder
        num_layers (int): Number of LSTM layers to use in the decoder
        num_classes (int): Number of output classes to predict
      
      Inputs:
        x (Tensor): (batch_size, time, d_encoder)
      
      Outputs:
        Tensor (batch_size, time, num_classes): Class prediction logits
    
    '''
    def __init__(self, d_encoder=144, d_decoder=320, num_layers=1, num_classes=29):
        super(LSTMDecoder, self).__init__()

        self.lstm = nn.LSTM(input_size=d_encoder, hidden_size=d_decoder, num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(d_decoder, num_classes)

    def forward(self, x):
        x, _ = self.lstm(x)
        logits = self.linear(x)
        return logits

class GreedyCharacterDecoder(nn.Module):
    ''' Greedy CTC decoder - Argmax logits and remove duplicates. '''
    def __init__(self):
        super(GreedyCharacterDecoder, self).__init__()

    def forward(self, x):
        indices = torch.argmax(x, dim=-1)
        indices = torch.unique_consecutive(indices, dim=-1)
        return indices.tolist()

class TransformerLrScheduler():
    '''
      Transformer LR scheduler from "Attention is all you need." https://arxiv.org/abs/1706.03762
      multiplier and warmup_steps taken from conformer paper: https://arxiv.org/abs/2005.08100
    '''
    def __init__(self, optimizer, d_model, warmup_steps, multiplier=5):
        self._optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.n_steps = 0
        self.multiplier = multiplier

    def step(self):
        self.n_steps += 1
        lr = self._get_lr()
        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr

    def _get_lr(self):
        return self.multiplier * (self.d_model ** -0.5) * min(self.n_steps ** (-0.5), self.n_steps * (self.warmup_steps ** (-1.5))) 

def model_size(model, name):
    ''' Print model size in num_params and MB'''
    param_size = 0
    num_params = 0
    for param in model.parameters():
        num_params += param.nelement()
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        num_params += buffer.nelement()
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print(f'{name} - num_params: {round(num_params / 1000000, 2)}M,  size: {round(size_all_mb, 2)}MB')

class TextTransform:
  ''' Map characters to integers and vice versa '''
  def __init__(self):
    self.char_map = {}
    for i, char in enumerate(range(65, 91)):
        self.char_map[chr(char)] = i
    self.char_map["'"] = 26
    self.char_map[' '] = 27
    self.index_map = {}

    for char, i in self.char_map.items():
        self.index_map[i] = char

  def text_to_int(self, text):
      ''' Map text string to an integer sequence '''
      int_sequence = []
      for c in text:
        ch = self.char_map[c]
        int_sequence.append(ch)
      return int_sequence

  def int_to_text(self, labels):
      ''' Map integer sequence to text string '''
      string = []
      for i in labels:
          if i == 28: # blank char
            continue
          else:
            string.append(self.index_map[i])
      return ''.join(string)

def get_audio_transforms():
  
  #  10 time masks with p=0.05
  #  The actual conformer paper uses a variable time_mask_param based on the length of each utterance.
  #  For simplicity, we approximate it with just a fixed value.
  time_masks = [torchaudio.transforms.TimeMasking(time_mask_param=15, p=0.05) for _ in range(10)]
  train_audio_transform = nn.Sequential(
    torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=80, hop_length=160), #80 filter banks, 25ms window size, 10ms hop
    torchaudio.transforms.FrequencyMasking(freq_mask_param=27),
    *time_masks,
  )

  valid_audio_transform = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=80, hop_length=160)

  return train_audio_transform, valid_audio_transform

def preprocess_example(data, data_type="train"):
  ''' Process raw LibriSpeech examples '''
  text_transform = TextTransform()
  train_audio_transform, valid_audio_transform = get_audio_transforms()
  spectrograms = []
  labels = []
  references = []
  input_lengths = []
  label_lengths = []

  for (waveform, _, utterance, _, _, _) in data:
      # Generate spectrogram for model input
      if data_type == 'train':
        spec = train_audio_transform(waveform).squeeze(0).transpose(0, 1) # (1, time, freq)
      else:
        spec = valid_audio_transform(waveform).squeeze(0).transpose(0, 1) # (1, time, freq)
      spectrograms.append(spec)

      # Labels 
      references.append(utterance) # Actual Sentence
      label = torch.Tensor(text_transform.text_to_int(utterance)) # Integer representation of sentence
      labels.append(label)

      # Lengths (time)
      input_lengths.append(((spec.shape[0] - 1) // 2 - 1) // 2) # account for subsampling of time dimension
      label_lengths.append(len(label))

  # Pad batch to length of longest sample
  spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True)
  labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)

  # Padding mask (batch_size, time, time)
  mask = torch.ones(spectrograms.shape[0], spectrograms.shape[1], spectrograms.shape[1])
  for i, l in enumerate(input_lengths):
      mask[i, :, :l] = 0

  return spectrograms, labels, input_lengths, label_lengths, references, mask.bool()

class BatchSampler(object):
    ''' Sample contiguous, sorted indices. Leads to less padding and faster training. '''
    def __init__(self, sorted_inds, batch_size):
        self.sorted_inds = sorted_inds
        self.batch_size = batch_size
    
    def __iter__(self):
        inds = self.sorted_inds.copy()

        while len(inds):
            to_take = min(self.batch_size, len(inds))
            start_ind = random.randint(0, len(inds) - to_take)
            batch_inds = inds[start_ind:start_ind + to_take]
            del inds[start_ind:start_ind + to_take]
            yield batch_inds

def main():
    if not os.path.isdir('./data'):
        os.makedirs('./data')
    train_data = LIBRISPEECH('./data', url='train-clean-100', download=True)
    test_data = LIBRISPEECH('./data', url='test-clean', download=True)

    sorted_train_ids = [idx for idx, _ in sorted(enumerate(train_data), key=lambda x: x[1][1])]
    sorted_test_ids = [idx for idx, _ in sorted(enumerate(test_data), key=lambda x: x[1][1])]

    train_loader = DataLoader(
        dataset=train_data,
        batch_sampler=BatchSampler(sorted_train_ids, batch_size=4),
        collate_fn=lambda x: preprocess_example(x, 'train')
        )



    test_loader = DataLoader(
        dataset=test_data,
        batch_sampler=BatchSampler(sorted_test_ids, batch_size=16),
        collate_fn=lambda x: preprocess_example(x, 'test')
        )

    encoder = ConformerEncoder(
                          d_input=80,
                          d_model=144,
                          num_layers=16,
                          conv_kernel_size=31, 
                          dropout=0.1,
                          feed_forward_residual_factor=0.5,
                          feed_forward_expansion_factor=4,
                          num_heads=4)
      
    decoder = LSTMDecoder(
                    d_encoder=144, 
                    d_decoder=320, 
                    num_layers=1)
    char_decoder = GreedyCharacterDecoder().eval()
    criterion = nn.CTCLoss(blank=28, zero_infinity=True)
    optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(decoder.parameters()), lr=5e-4, betas=(.9, .98), eps=1e-05 if False else 1e-09, weight_decay=1e-6)
    scheduler = TransformerLrScheduler(optimizer, 144, 10000)

    # Print model size
    model_size(encoder, 'Encoder')
    model_size(decoder, 'Decoder')

    gc.collect()

    # GPU Setup
    if torch.cuda.is_available():
      print('Using GPU')
      gpu = True
      torch.cuda.set_device('cuda:3')
      criterion = criterion.cuda()
      encoder = encoder.cuda()
      decoder = decoder.cuda()
      char_decoder = char_decoder.cuda()
      torch.cuda.empty_cache()
    else:
      gpu = False

    # Mixed Precision Setup
    #if False:
    #    print('Using Mixed Precision')
    grad_scaler = GradScaler(enabled=False)

    # Initialize Checkpoint 
    #if args.load_checkpoint:
    #    start_epoch, best_loss = load_checkpoint(encoder, decoder, optimizer, scheduler, args.checkpoint_path)
    #  print(f'Resuming training from checkpoint starting at epoch {start_epoch}.')
    #else:
    start_epoch = 0
    best_loss = float('inf')

    # Train Loop
    optimizer.zero_grad()
    for epoch in range(start_epoch, 50):
        torch.cuda.empty_cache()

        #variational noise for regularization
        add_model_noise(encoder, std=0.0001, gpu=gpu)
        add_model_noise(decoder, std=0.0001, gpu=gpu)

        # Train/Validation loops
        wer, loss = train(encoder, decoder, char_decoder, optimizer, scheduler, criterion, grad_scaler, train_loader, gpu=gpu) 
        valid_wer, valid_loss = validate(encoder, decoder, char_decoder, criterion, test_loader, gpu=gpu)
        print(f'Epoch {epoch} - Valid WER: {valid_wer}%, Valid Loss: {valid_loss}, Train WER: {wer}%, Train Loss: {loss}')  

        # Save checkpoint 
        if valid_loss <= best_loss:
            print('Validation loss improved, saving checkpoint.')
            best_loss = valid_loss
            #save_checkpoint(encoder, decoder, optimizer, scheduler, valid_loss, epoch+1, args.checkpoint_path)

if __name__ == '__main__':
    main() 