import torch
import torchaudio
import sentencepiece as spm
import os

import pandas as pd
import csv

class Subword():
  def __init__(self, data_train, data_test):
    self.data_train = data_train
    self.data_test = data_test
    self.sp = spm.SentencePieceProcessor()
    self.dir = './subword'

  def make_txt(self):

    if not os.path.exists(self.dir):
        os.makedirs(self.dir)

    libri_utterance = []

    for (_, _, utterance_train, _, _, _), (_, _, utterance_test, _, _, _)  in zip(self.data_train[0],self.data_test[0]):
      libri_utterance.append(utterance_train)
      libri_utterance.append(utterance_test)

    with open('./subword/subword_libri.txt', 'w', encoding='utf-8') as file:
      for sentence in libri_utterance:
        file.write(sentence + '\n')

    spm.SentencePieceTrainer.Train(
      '--input=subword/subword_libri.txt --model_prefix=SentencePiece_bpe --vocab_size=1000 --model_type=bpe --max_sentence_length=9999')

    spm.SentencePieceTrainer.Train(
      '--input=subword/subword_libri.txt --model_prefix=SentencePiece_ngram --vocab_size=1000 --model_type=unigram --max_sentence_length=9999')

    vocab_file = "SentencePiece_bpe.model"
    self.sp.load(vocab_file)

    min_id = 0
    max_id = self.sp.GetPieceSize() - 1

    print("최소 ID:", min_id)
    print("최대 ID:", max_id)

    vocab_file = "SentencePiece_ngram.model"
    self.sp.load(vocab_file)

    min_id = 0
    max_id = self.sp.GetPieceSize() - 1

    print("최소 ID:", min_id)
    print("최대 ID:", max_id)

  #bpe
  def SentencePiece_bpe_toInt(self, utterance):

    vocab_file = "SentencePiece_bpe.model"
    self.sp.load(vocab_file)

    return self.sp.encode_as_ids(utterance)

  def SentencePiece_bpe_toStr(self, int_list):

    vocab_file = "SentencePiece_bpe.model"
    self.sp.load(vocab_file)

    blank_id = self.sp.GetPieceSize()

    # Remove the blank ID from the int_list
    filtered_int_list = [id for id in int_list if id != blank_id]

    # Decode the filtered list of IDs
    return self.sp.DecodeIds(filtered_int_list)

  #n_gram
  def SentencePiece_ngram_toInt(self, utterance):

    vocab_file = "SentencePiece_ngram.model"
    self.sp.load(vocab_file)

    return self.sp.encode_as_ids(utterance)

  def SentencePiece_ngram_toStr(self, int_list):

    vocab_file = "SentencePiece_ngram.model"
    self.sp.load(vocab_file)

    blank_id = self.sp.GetPieceSize()

    # Remove the blank ID from the int_list
    filtered_int_list = [id for id in int_list if id != blank_id]

    # Decode the filtered list of IDs
    return self.sp.DecodeIds(filtered_int_list)





'''
libri_utterance = []
for (waveform, _, utterance, _, _, _) in data_train[0]:
    libri_utterance.append(utterance)

libri_path = Path(__file__).parent / f"/data/libri_utterance.txt"

with open('./data/libri_utterance.txt', 'w', encoding='utf-8') as file:
    for sentence in libri_utterance:
        file.write(sentence + '\n')

spm.SentencePieceTrainer.Train('--input=data/libri_utterance.txt --model_prefix=imdb --vocab_size=5000 --model_type=bpe --max_sentence_length=9999')
'''
'''
sp = spm.SentencePieceProcessor()
vocab_file = "imdb.model"
sp.load(vocab_file)

lines = [
  "CHAPTER ONE MISSUS RACHEL LYNDE IS SURPRISED MISSUS RACHEL LYNDE LIVED JUST WHERE THE AVONLEA MAIN ROAD DIPPED DOWN INTO A LITTLE HOLLOW FRINGED WITH ALDERS AND LADIES EARDROPS AND TRAVERSED BY A BROOK",
  "THE IDENTITY OF THE FINAL VICTIM"
]
for line in lines:
  print(line)
  print(sp.encode_as_pieces(line))
  print(sp.encode_as_ids(line))
  print()

print(sp.GetPieceSize())
print(sp.IdToPiece(430))
print(sp.PieceToId('▁character'))
print(sp.DecodeIds([41, 141, 1364, 1120, 4, 666, 285, 92, 1078, 33, 91]))
print(sp.encode('I have waited a long time for someone to film', out_type=str))
print(sp.encode('I have waited a long time for someone to film', out_type=int))
'''
'''
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

# 토크나이저와 트레이너 초기화
tokenizer = Tokenizer(BPE())
trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])

# 토크나이저 훈련
tokenizer.train(["data/libri_utterance.txt"], trainer)

# 텍스트 토크나이징
text = "I MUST HAVE BEEN ASLEEP A VERY LONG TIME THEN HIS SON EXPLAINED TO HIM ALL THAT HAD HAPPENED AND GAVE HIM SOME FOOD"
output = tokenizer.encode(text)

# 토큰과 정수 인덱스 출력
print("Tokens:", output.tokens)
print("Token IDs:", output.ids)


# 토크나이저 사용 예시
output = tokenizer.encode("Example text to encode")
print(output.tokens)
'''
'''
#wordpiece
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer

tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
trainer = WordPieceTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
tokenizer.train(["data/libri_utterance.txt"], trainer)

text = "AND AMONG THEM WAS ONE WHICH HAD THE DRIED UP BODY OF THE BOY'S FATHER STUCK ON ITS HORN THE BOY WAS RATHER FRIGHTENED AND SANG"
output = tokenizer.encode(text)
print("Tokens:", output.tokens)
print("Token IDs:", output.ids)



# unigram sentencepiece
spm.SentencePieceTrainer.Train('--input=data/libri_utterance.txt --model_prefix=imdb2 --vocab_size=8000 --model_type=unigram')
'''
'''
sp = spm.SentencePieceProcessor()
vocab_file = "imdb2.model"
sp.load(vocab_file)

lines = [
  "CHAPTER ONE MISSUS RACHEL LYNDE IS SURPRISED MISSUS RACHEL LYNDE LIVED JUST WHERE THE AVONLEA MAIN ROAD DIPPED DOWN INTO A LITTLE HOLLOW FRINGED WITH ALDERS AND LADIES EARDROPS AND TRAVERSED BY A BROOK",
  "THE IDENTITY OF THE FINAL VICTIM"
]
for line in lines:
  print(line)
  print(sp.encode_as_pieces(line))
  print(sp.encode_as_ids(line))
  print()

print(sp.GetPieceSize())
print(sp.IdToPiece(1))
print(sp.PieceToId('_'))
print(sp.DecodeIds([41, 141, 1364, 1120, 4, 666, 285, 92, 1078, 33, 91]))
print(sp.encode('I MUST HAVE BEEN ASLEEP A VERY LONG TIME THEN HIS SON EXPLAINED TO HIM ALL THAT HAD HAPPENED AND GAVE HIM SOME FOOD', out_type=str))
print(sp.encode('I MUST HAVE BEEN ASLEEP A VERY LONG TIME THEN HIS SON EXPLAINED TO HIM ALL THAT HAD HAPPENED AND GAVE HIM SOME FOOD', out_type=int))

'''