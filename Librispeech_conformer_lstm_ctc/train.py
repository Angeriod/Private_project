from copy import deepcopy
import logging

import numpy as np
from tqdm.auto import tqdm
import torch
import torch.nn.functional as F
import wandb
import random
import os
import metrics
import torchmetrics.functional as FNT

from torch.cuda.amp import autocast, GradScaler
from metrics import WER
from sklearn.metrics import classification_report, confusion_matrix

from torchmetrics.text.wer import WordErrorRate
from utils import mat_To_device,TextTransform, add_model_noise


class Trainer:
    def __init__(self, encoder, decoder, char_decoder, optimizer, loss_function, scheduler,grad_scaler, config, subword):
        super(Trainer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.char_decoder = char_decoder
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.scheduler = scheduler
        self.grad_scaler = grad_scaler
        self.config = config
        self.subword = subword
        pass

    def train_step(self, data, epoch, device):
        train_wer = 0.
        train_loss = 0.
        #train_f1 = 0.
        count = 0
        text_transform = TextTransform()
        wer = WordErrorRate()
        target_list=[]
        pred_list = []

        self.encoder.train()
        self.decoder.train()

        with torch.set_grad_enabled(True):
            with tqdm(total=len(data), desc=f"EPOCH - {epoch} ") as pbar:
                for step, (batch) in enumerate(data):



                    count += 1
                    spectrograms, labels, input_lengths, label_lengths, references, mask = batch
                    #print(f'{spectrograms.shape} {labels} {input_lengths} {label_lengths} {references} {mask}')
                    #cuda
                    spectrograms = mat_To_device(spectrograms,device)
                    labels = mat_To_device(labels,device)
                    input_lengths = mat_To_device(torch.tensor(input_lengths),device)
                    label_lengths = mat_To_device(torch.tensor(label_lengths),device)
                    mask = mat_To_device(mask, device)

                    #print(f'spectrograms: {spectrograms} labels:{labels} input_lengths:{input_lengths} label_lengths:{label_lengths} mask:{mask}')

                    with autocast(enabled=False):
                        y_pred = self.encoder(spectrograms, mask)
                        outputs = self.decoder(y_pred)
                        loss = self.loss_function(F.log_softmax(outputs, dim=-1).transpose(0,1), labels, input_lengths, label_lengths)

                    self.grad_scaler.scale(loss).backward()
                    self.grad_scaler.step(self.optimizer)
                    self.grad_scaler.update()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                    inds = self.char_decoder(outputs.detach())

                    for sample in inds:
                        if self.config.subword == 'ASCII':
                            pred_list.append(text_transform.int_to_text(sample))
                        elif self.config.subword == 'BPE':
                            pred_list.append(self.subword.SentencePiece_bpe_toStr(sample))
                        else:
                            pred_list.append(self.subword.SentencePiece_ngram_toStr(sample))

                    if count % 1000 ==0:
                        print(f'pred_list: {pred_list}')

                    #self.optimizer.zero_grad()

                    #self.optimizer.step()

                    #train_accuracy += FNT.classification.binary_accuracy(y_pred, y)
                    #train_f1 += FNT.classification.binary_f1_score(y_pred, y)

                    train_loss += loss.detach().cpu()



                    learning_rate = self.scheduler._get_lr()

                    train_wer += wer(pred_list, references) * 100
                    pbar.update(1)
                    pbar.set_postfix_str(
                        f"Train WER: {train_wer/count:.2f}  Loss: {loss.item():.4f} ({train_loss / count:.4f}) Learning Rate: {learning_rate:.2e}"
                    )

               # cm = confusion_matrix(target_list, pred_list) Train f1: {(train_f1 / count):1.4f}
               # print('\n' + str(cm))

            return train_wer / count,  train_loss / count, learning_rate

    def validation_step(self, data, epoch, device):
        validation_wer = 0.
        #validation_f1 = 0.
        validation_loss = 0.
        validation_count = 0
        text_transform = TextTransform()
        wer = WordErrorRate()
        pred_list = []

        self.encoder.eval()
        self.decoder.eval()

        with torch.no_grad():
            with tqdm(total=len(data), desc=f"EPOCH - {epoch} ") as pbar:
                for step, (batch) in enumerate(data):
                    validation_count += 1

                    spectrograms, labels, input_lengths, label_lengths, references, mask = batch

                    # cuda
                    spectrograms = mat_To_device(spectrograms, device)
                    labels = mat_To_device(labels, device)
                    input_lengths = mat_To_device(torch.tensor(input_lengths),device)
                    label_lengths = mat_To_device(torch.tensor(label_lengths),device)
                    mask = mat_To_device(mask, device)

                    with autocast(enabled=False):
                        outputs = self.encoder(spectrograms,mask)
                        outputs = self.decoder(outputs)
                        loss = self.loss_function(F.log_softmax(outputs, dim=-1).transpose(0,1), labels, input_lengths, label_lengths)
                    validation_loss += loss.detach().cpu()

                    inds = self.char_decoder(outputs.detach())

                    for sample in inds:
                        if self.config.subword == 'ASCII':
                            pred_list.append(text_transform.int_to_text(sample))
                        elif self.config.subword == 'BPE':
                            pred_list.append(self.subword.SentencePiece_bpe_toStr(sample))
                        else:
                            pred_list.append(self.subword.SentencePiece_ngram_toStr(sample))

                    #validation_accuracy += FNT.classification.binary_accuracy(y_pred, y)
                    #validation_f1 += FNT.classification.binary_f1_score(y_pred, y)
                    validation_wer += wer(pred_list, references) * 100

                    pbar.update(1)
                    pbar.set_postfix_str(
                        f"valid WER: {validation_wer/validation_count:.2f} Loss: {loss.item():.4f} ({validation_loss / validation_count:.4f})"
                    )
                    
                #cm = confusion_matrix(target_list, pred_list)
                #print('\n' + str(cm))
                   
        return validation_wer/validation_count, validation_loss/validation_count

    def fit(self, train_dataloader, test_dataloader, device):
        best_model = None
        best_score = 0.

        self.optimizer.zero_grad()

        for epoch in range(self.config.n_epochs):

            train_wer, train_loss, learning_rate = self.train_step(train_dataloader, epoch, device)
            valid_wer, valid_loss = self.validation_step(test_dataloader, epoch, device)

            print(f'Epoch:{epoch}|| train_wer: {train_wer}| train_loss:{train_loss} | valid_wer:{valid_wer} | valid_loss:{valid_loss}')

            if valid_wer >= best_score:
                best_model = deepcopy(self.encoder.state_dict())
                print(f"SAVE! Epoch: {epoch + 1}/{self.config.n_epochs}")
                best_score = valid_wer

                '''
                # save model
                if not os.path.exists(self.config.model_dir):
                    os.makedirs(self.config.model_dir)


                model_name = "epoch{}-loss{:.4f}-wer{:.4f}.pt".format(epoch, valid_loss, valid_wer)
                model_path = os.path.join(self.config.model_dir, model_name)
                torch.save({
                    'model': self.encoder.state_dict(),
                    'config': self.config
                }, model_path)

            if self.config.wan >= 1:
                wandb.log({
                    "Train loss": train_loss,
                    "Train WER": train_wer,
                    #"Train F1-score": train_f1,
                    "Learning rate": learning_rate,
                    "Validation loss": valid_loss,
                    "Validation WER": valid_wer,
                    #"Validation F1-score": valid_f1
                })
            '''
        self.encoder.load_state_dict(best_model)
