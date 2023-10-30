# Private_project
## 1.implementing Conformer paper
#### Dataset : Librispeech 100h
#### Preprocessing : followed all paper defaults and text was Integer encoded to aligned waveform 
#### model : Conformer-L with CTCloss
#### train: 20 epoch with RTX 3060ti, others: paper default
#### result : overall Train wer -> about 50  // overall Valid wer -> about 83
#### future application: Thinking about low WER was caused by a bunch of silent in single waveform -> Using VAD (silero vad) to delete silent period 
