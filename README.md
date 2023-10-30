# Private_project
#### Dataset : Librispeech 100h
#### Preprocessing : Integer encoding to aligned waveform 
#### model : Conformer-L with CTCloss
#### train: 20 epoch with RTX 3060ti
#### result : overall Train wer -> about 50%  // overall Valid wer -> about 83%
#### future application: Thinking about low WER was caused by a bunch of silent in single waveform -> Using VAD (silero vad) to delete silent period 
