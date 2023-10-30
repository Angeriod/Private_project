import glob
import os
import re
import textgrids

import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T

def phone_alignment(corpus):
    phones_alignment=[]
    for phones in corpus:
        alignment = textgrids.TextGrid(phones)
        alignment = alignment.interval_tier_to_array('phones')
        phones_alignment.append(alignment)

    return phones_alignment

def extract_audio(audio_paths):
    audio_list = []
    for path in audio_paths:
        waveform , _ = torchaudio.load(path)
        audio_list.append(waveform)

    return audio_list
 
def Mel_spectrogram(audio_chunk_list,config):
    
    mel_spec_list= []
    for audio_chunk in audio_chunk_list:
        sample_rate=config.sample_rate
        n_fft = int(0.025 * config.sample_rate)
        hop_length = int(0.01 * config.sample_rate)
        n_mels = config.n_mels

        mel_audio_chunk = T.MelSpectrogram(
        sample_rate = sample_rate,
        n_fft = n_fft,
        hop_length = hop_length,
        n_mels=n_mels,
        )
        
        melspec = mel_audio_chunk(audio_chunk) 
        mel_spec_list.append(melspec)
        
    return mel_spec_list  

def assign_label_length(phone_alignment : list, label_dictionary : dict ): 

    labels = []
    length = []

    for phones_alignment in phone_alignment:
            
            idx_label = []
            idx_length = []

            for phone_alignment_sub in phones_alignment:
                label = phone_alignment_sub['label']
                label = label_dictionary[label]
                start = phone_alignment_sub['begin']
                end = phone_alignment_sub['end']
                length_label = round((end-start)/ 0.01)
                idx_label.append(label)
                idx_length.append(length_label)

            labels.append(idx_label)
            length.append(idx_length)
    
    return labels, length

def make_dictionary(phone_alignments):
    phone_list=[]
    
    for phone_alignment in phone_alignments:
        for phone in phone_alignment:
            phone_list.append(phone['label'])

    
    char_to_idx = {char: idx+1 for idx, char in enumerate(set(phone_list)) if char != ""}
    char_to_idx[""]=0
    
    return char_to_idx

