# Project Overview

This repository is dedicated to exploring advanced speech recognition techniques using the LibriSpeech dataset. We've implemented various models and preprocessing methods from scratch, aiming to improve the word error rate (WER) significantly.

## Experiments

### Experiment 1: LibriSpeech Phoneme TextGrid

- **Dataset**: LibriSpeech 100h + dev clean
- **Preprocessing**: Adhered to paper defaults; text was integer-encoded to align with the waveform. It's important to note that alignment is not required due to the utilization of CTC loss.
- **Model**: Conformer-L with CTC loss
- **Training**: Conducted over 20 epochs on an RTX 3060 Ti, with other settings following paper defaults
- **Results**: Achieved a Train WER of ~50% and a Valid WER of ~83%
- **Future Consideration**: Suspected that low WER is due to silence within waveforms. Considering the use of VAD (e.g., Silero VAD) to remove silent periods.
- **Conclusion**: Encountered issues with NaN output values; this issue was not present with files from torchaudio, leading to the discontinuation of this experiment.

### Experiment 2: LibriSpeech Conformer LSTM CTC

- **Dataset**: LibriSpeech clean 100h + dev clean
- **Waveform Preprocessing**: Followed paper defaults; text was integer-encoded to align with the waveform. Attempts to use VAD did not yield performance improvements. Notably, using Silero VAD significantly increased the time required to create the dataset.
- **Tokenizer**: Utilized integer indexing (using ASCII for alphabets). Also experimented with SentencePiece for subword tokenization, which did not yield any performance improvements.
- **Model**: Conformer-small + LSTM with CTC loss
- **Training**: Extended over 50 epochs on an RTX 3090, adhering to paper defaults
- **Results**: No significant change in CTC performance observed. WER and CER stagnated around 100, with no further learning progression.
- **Future Consideration**: Due to the inability to improve performance, the project has been temporarily halted. Plans to revisit the project with enhanced skills and potentially new approaches are in consideration.

## Conclusion

Through this project, we really got into the nitty-gritty of making computers understand spoken words better, tackling tricky stuff like figuring out what to do with those quiet moments in recordings where nobody's talking. Even though it was tough at times, all the tests and tinkering we did have laid a pretty cool groundwork for anyone who wants to dive deeper into making speech recognition tech better in the future. We're super keen on coming back to the challenges we couldn't crack this time around. It's all about not giving up and trying to push past what we currently know how to do. We learned a ton, and we're pumped to see how far we can take this!
