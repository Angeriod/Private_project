import os
from torch.utils.data import Dataset, IterableDataset
import torch
import time
from utils import phone_alignment,assign_label_length,Mel_spectrogram,make_dictionary,extract_audio
from glob import glob
import argparse
from datetime import datetime

class CustomDataset(Dataset):
    def __init__(self, config, train):
        super(CustomDataset,self).__init__()
        self.config=config
        if train:
            self.audio_path = glob(self.config.data_dir+self.config.train_audio_dir+'/19/*/*.flac')
            self.align_path = glob(self.config.data_dir+self.config.train_corpus_dir+'/19/*/*.TextGrid')
        else:
            self.audio_path = glob(self.config.data_dir+self.config.test_audio_dir+'/61/*/*.flac')
            self.align_path = glob(self.config.data_dir+self.config.test_corpus_dir+'/61/*/*.TextGrid')

        self.audio_list = extract_audio(self.audio_path)
        self.phones_alignment = phone_alignment(self.align_path)
        self.letter_dictionary = make_dictionary(self.phones_alignment)
        self.label_index, self.labels_lengths = assign_label_length(self.phones_alignment,self.letter_dictionary)
        self.mel_audio_index = Mel_spectrogram(self.audio_list,self.config)

    def __len__(self):

        return len(self.phones_alignment)

    def __getitem__(self, idx):
        mel_spectrogram_idx = self.mel_audio_index[idx]
        label = self.label_index[idx]
        label_lengths = self.labels_lengths[idx]
        
        return mel_spectrogram_idx.squeeze(0), label, label_lengths
    
    def collate_fn(self, batch):

            mel_specs, labels, label_lengths = zip(*batch)
            
            max_length = max(mel_spec.size(1) for mel_spec in mel_specs)

            masked_specs = []
            masks = []
            lengths = []
            labels_list = []
            label_lengths = []
            label_max = max(len(i) for i in labels)

        
            for label,mel in zip(labels,mel_specs):
            
                    current_time_step = mel.size(1)
                    padding_length = max_length - current_time_step
                    padded_spec = torch.nn.functional.pad(mel, (0, padding_length)).permute(1,0)
                    mask = torch.ones(max_length)
                    mask[current_time_step:] = 0
                    if label_max >len(label):
                        label_lengths.append(len(label))
                        x = (label_max-len(label))
                        label.extend([0 for i in range(x)])
                        labels_list.append(label)
                    else:
                        label_lengths.append(len(label))
                        labels_list.append(label)

                    lengths.append(current_time_step)
                    masked_specs.append(padded_spec)
                    masks.append(mask)
            
            masked_specs = torch.stack(masked_specs)
            masks = torch.stack(masks)
            labels = torch.tensor(labels_list)
            label_lengths = torch.tensor(label_lengths)
            mel_lengths = torch.tensor(lengths)
        
            #label_length = torch.tensor(label_lengths)

            return masked_specs, mel_lengths, labels, label_lengths
     
    

