import os
from torch.utils.data import Dataset, IterableDataset
import torch
import time
from utils import phone_alignment,audio_chunk,Mel_spectrogram,make_dictionary,extract_audio
from glob import glob
import argparse
from datetime import datetime

class CustomDataset(Dataset):
    def __init__(self, config, train):
        super(CustomDataset,self).__init__()
        self.config=config
        if train:
            self.audio_path = glob(self.config.data_dir+self.config.train_audio_dir+'/103/*/*.flac')
            self.align_path = glob(self.config.data_dir+self.config.train_corpus_dir+'/103/*/*.TextGrid')
        else:
            self.audio_path = glob(self.config.data_dir+self.config.test_audio_dir+'/61/*/*.flac')
            self.align_path = glob(self.config.data_dir+self.config.test_corpus_dir+'/61/*/*.TextGrid')

        self.audio_list = extract_audio(self.audio_path)
        self.phones_alignment = phone_alignment(self.align_path)
        self.letter_dictionary = make_dictionary(self.phones_alignment)
        self.audio_chunk_ , self.labels = audio_chunk(self.audio_list,self.phones_alignment,self.config)
        self.mel_audio_chunk = Mel_spectrogram(self.audio_chunk_,self.config)

    def __len__(self):

        return len(self.phones_alignment)

    def __getitem__(self, idx):
        mel_spectrogram_idx = self.mel_audio_chunk[idx]
        labels_idx = self.labels[idx]
        
        labels = self.letter_dictionary[labels_idx] 
        
        return mel_spectrogram_idx.squeeze(0), [labels]
    
    def collate_fn(self, batch):

            mel_specs, labels = zip(*batch)
            max_length = max(mel_spec.size(1) for mel_spec in mel_specs)

            masked_specs = []
            masks = []
            lengths = []
            
            for mel in mel_specs:
            
                    current_time_step = mel.size(1)
                    padding_length = max_length - current_time_step
                    padded_spec = torch.nn.functional.pad(mel, (0, padding_length)).permute(1,0)
                    mask = torch.ones(max_length)
                    mask[current_time_step:] = 0

                    lengths.append(current_time_step)
                    masked_specs.append(padded_spec)
                    masks.append(mask)
            
            masked_specs = torch.stack(masked_specs)
            masks = torch.stack(masks)
            lengths = torch.tensor(lengths)
            labels = torch.tensor(labels) 
            return masked_specs, lengths, labels
     
    
'''
class MyIterableDataset(IterableDataset):
    def __init__(self, data):
        self.data = data

    def __iter__(self):
        for x in self.data:
            worker = torch.utils.data.get_worker_info()
            worker_id = worker.id if worker is not None else -1

            start = time.time()
            time.sleep(0.1)
            end = time.time()

            yield x, worker_id, start, end
'''



# Iterable dataset의 경우 직접 worker 별로 일을 재분배 해야함
def worker_init_fn():
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    worker_id = worker_info.id
    split_size = len(dataset.data) // worker_info.num_workers
    dataset.data = dataset.data[worker_id * split_size: (worker_id + 1) * split_size]
