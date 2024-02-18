import glob
import os
import re

from torch import Tensor
from typing import List,Tuple,Dict,Union
import argparse
from tqdm import tqdm

import torch
import torchaudio
import torch.nn as nn
import torchaudio.functional as F
import torchaudio.transforms as T
import random
from subword import Subword
import math

def model_size(model, name):
    param_size = 0
    num_params = 0
    for param in model.parameters():
        num_params +=param.nelement()
        param_size +=param.nelement() * param.element_size()

    buffer_size = 0

    for buffer in model.buffers():
        num_params +=buffer.nelement()
        buffer_size +=buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print(f'{name} - num_params: {round(num_params / 1000000, 2)}M,  size: {round(size_all_mb, 2)}MB')


class BatchSampler(object):
    def __init__(self, sorted_inds, batch_size):
        self.sorted_inds = sorted_inds
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.sorted_inds) / self.batch_size)

    def __iter__(self):
        inds = self.sorted_inds.copy()

        while len(inds):
            to_take = min(self.batch_size, len(inds))

            start_ind = random.randint(0, len(inds) - to_take)
            batch_inds = inds[start_ind:start_ind + to_take]
            del inds[start_ind:start_ind + to_take]
            yield batch_inds


class TextTransform:
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
      int_sequence = []
      for c in text:
        ch = self.char_map[c]
        int_sequence.append(ch)
      return int_sequence

  def int_to_text(self, labels):
      string = []
      for i in labels:
          if i == 28: # blank char
            continue
          else:
            string.append(self.index_map[i])
      return ''.join(string)

def get_audio_transforms(config):

    time_masks = [T.TimeMasking(time_mask_param=config.time_mask_param, p=config.time_mask_param_p) for _ in range(10)]
    train_audio_transform = nn.Sequential(
        T.MelSpectrogram(sample_rate=config.sample_rate,n_mels=config.n_mels,n_fft=config.n_fft,hop_length=config.hop_length),
        T.FrequencyMasking(freq_mask_param=config.freq_mask_param),
        *time_masks,
    )

    valid_audio_transform = T.MelSpectrogram(sample_rate=config.sample_rate,n_mels=config.n_mels,n_fft=config.n_fft,hop_length=config.hop_length)

    return train_audio_transform, valid_audio_transform


def preprocess_example(data, config, subword_loader, data_type='train',mode='ASCII' ):
    text_transform = TextTransform()
    train_audio_transform, valid_audio_transform = get_audio_transforms(config)
    spectrograms = []
    labels = []
    references = []
    input_lengths = []
    label_lengths = []

    for (waveform, _, utterance, _, _, _) in data:
        # spectrogram
        if data_type == 'train':
            spec = train_audio_transform(waveform).squeeze(0).transpose(0 ,1)
        else:
            spec = valid_audio_transform(waveform).squeeze(0).transpose(0, 1)

        spectrograms.append(spec)

        #labels
        references.append(utterance)

        if mode == 'ASCII':
            label = torch.Tensor(text_transform.text_to_int(utterance))
        elif mode == 'BPE':
            label = torch.Tensor(subword_loader.SentencePiece_bpe_toInt(utterance))
        elif mode == 'NGRAM':
            label = torch.Tensor(subword_loader.SentencePiece_ngram_toInt(utterance))

        labels.append(label)

        #lengths (time)
        input_lengths.append(((spec.shape[0] - 1) // 2 - 1) //2 )
        label_lengths.append(len(label))

    # zero padding
    spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True)
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)

    #mask
    mask = torch.ones(spectrograms.shape[0], spectrograms.shape[1], spectrograms.shape[1])
    for i, l in enumerate(input_lengths):
        mask[i, :, :l] = 0

    return spectrograms, labels, input_lengths, label_lengths, references, mask.bool()


def audio_VAD_(path: str, config: argparse.Namespace, VAD_model, utils) -> Tensor:
    """
    이 함수는 주어진 오디오 파일 경로에 대해 VAD(음성 활동 감지) 처리를 수행합니다.

    오디오 파일은 torchaudio를 사용하여 로드되며, VAD 모델은 음성 신호에서 음성이 있는 부분을 탐지합니다.
    VAD 처리는 utils 모듈의 함수들을 사용하여 수행됩니다.

    매개변수:
    - path (str): 처리할 오디오 파일의 경로.
    - config (argparse.Namespace): 오디오 처리에 사용될 설정을 포함하는 객체.
    - VAD_model: VAD 처리에 사용될 모델.
    - utils: 오디오 처리에 사용될 유틸리티 함수들.

    반환값:
    - Tensor: VAD 처리가 완료된 오디오 데이터의 Tensor.
    """

    # utils 모듈에서 필요한 함수들을 로드
    (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils

    # torchaudio를 사용하여 오디오 파일 로드
    wav, _ = torchaudio.load(path)

    # 음성 신호에서 음성이 있는 부분을 탐지
    speech_timestamps = get_speech_timestamps(wav.squeeze(0), VAD_model, sampling_rate=config.sample_rate)

    # 음성이 있는 부분만을 수집하여 Tensor로 반환
    return collect_chunks(speech_timestamps, wav.squeeze(0)).unsqueeze(0).requires_grad_(True)

def add_model_noise(model, device, std=0.0001):
    with torch.no_grad():
        for param in model.parameters():
            param.add_(torch.randn(param.size()).to(device) *std)

class TransformerLrScheduler():
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
def mat_To_device(x : Union[List[Tensor], Tuple[Tensor, ...], Tensor], device: torch.device) -> Union[List[Tensor], Tensor]:

    """
    이 함수는 Tensor 또는 Tensor들의 집합(리스트 또는 튜플 형태)을 입력받아,
    주어진 PyTorch 디바이스로 이동시키는 역할을 합니다.
    이는 모델을 GPU나 다른 하드웨어 가속기로 옮길 때 유용합니다.

    매개변수:
    - x (Union[List[Tensor], Tuple[Tensor, ...], Tensor]): 이동시킬 Tensor 또는 Tensor들의 집합.
      단일 Tensor 또는 Tensor의 리스트 또는 튜플이 될 수 있습니다.
    - device (torch.device): Tensor들을 옮길 대상 디바이스.

    반환값:
    - Union[List[Tensor], Tensor]: 디바이스로 이동된 Tensor 또는 Tensor 리스트.
      입력이 단일 Tensor이면 단일 Tensor를, 리스트나 튜플이면 리스트 형태로 반환합니다.
    """

    if isinstance(x, (tuple, list)):
        # 입력 `x`가 튜플이나 리스트인 경우:
        # `x`의 각 요소를 반복하며 각각을 `device`로 이동시킵니다.
        return [x_.to(device) for x_ in x]  # 변환된 Tensor 리스트를 반환합니다.
    else:
        # 입력 `x`가 단일 Tensor인 경우:
        # 해당 Tensor를 `device`로 이동시킵니다.
        return x.to(device)  # 변환된 단일 Tensor를 리스트로 묶어 반환합니다.



