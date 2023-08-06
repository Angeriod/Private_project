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

def audio_chunk(audio_list,phones_alignment,config):

    sample_rate = config.sample_rate
    audio_list_chunk = []
    labels=[]

    for audio, phones_alignment in zip(audio_list,phones_alignment):
        for phone_alignment_sub in phones_alignment:
            label = phone_alignment_sub['label']
            start = int(phone_alignment_sub['begin'] * sample_rate) 
            end = int(phone_alignment_sub['end'] * sample_rate)

            segment = audio[:,start:end]
            audio_list_chunk.append(segment)
            labels.append(label)
            


    return audio_list_chunk, labels

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
        n_fft = config.n_fft
        win_length = config.win_length
        hop_length = config.hop_length
        n_mels = config.n_mels

        mel_audio_chunk = T.MelSpectrogram(
        sample_rate = sample_rate,
        n_fft = n_fft,
        win_length = win_length,
        hop_length = hop_length,
        n_mels=n_mels,
        )
        
        melspec = mel_audio_chunk(audio_chunk) 
        mel_spec_list.append(melspec)
        
    return mel_spec_list  

def make_dictionary(phone_alignments):
    phone_list=[]
    
    for phone_alignment in phone_alignments:
        for phone in phone_alignment:
            phone_list.append(phone['label'])

    
    char_to_idx = {char: idx+1 for idx, char in enumerate(set(phone_list))}
    char_to_idx[0]=0
    

    return char_to_idx

