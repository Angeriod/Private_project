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

from glob import glob
import metrics
from sklearn.metrics import classification_report, confusion_matrix
from utils import phone_alignment,make_dictionary
from metrics import WER

class Trainer:
    def __init__(self, model, optimizer, loss_function, scheduler, config):
        super(Trainer, self).__init__()
        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.scheduler = scheduler
        self.config = config

        pass

    def train_step(self, data, epoch, device):
        train_wer = 0.
        train_loss = 0.
        #train_f1 = 0.
        count = 0
        target_list=[]
        pred_list=[]

        with tqdm(total=len(data), desc=f"EPOCH - {epoch} ") as pbar:
            for step, (x, x_L, y, y_L) in enumerate(data):
                # 
                self.model.train()
                self.model.zero_grad()
                

                count += 1
                if isinstance(x, tuple) or isinstance(x, list):
                    x = [x_.to(device) for x_ in x]
                else:
                    x = x.to(device)
                
                if isinstance(y, tuple) or isinstance(y, list):
                    y = [y_.to(device) for y_ in y]
                else:
                    y = y.to(device)

                if isinstance(x_L, tuple) or isinstance(x_L, list):
                    x_L = [x_L_.to(device) for x_L_ in x_L]
                else:
                    x_L = x_L.to(device)

                if isinstance(y_L, tuple) or isinstance(y_L, list):
                    y_L = [y_L_.to(device) for y_L_ in y_L]
                else:
                    y_L = y_L.to(device)

                y_pred, y_pred_lengths = self.model(x,x_L)
                '''
                x : mel spectrogram 배치씩의 각 mel spectrogram들 제로패딩 있음
                x_L : mel spectrogram의 실제 각각의 길이(패딩제외)
                y: 각 target들, 제로 패딩 들어감: 무음처리-> ""로 인덱싱
                y_L : target의 제로 패딩이 없는 실제 길이
                '''
    
                loss = self.loss_function(y_pred.transpose(0,1), y, y_pred_lengths, y_L)
                
                if torch.isnan(loss).any():
                    print("NaN이 발생했습니다.")
                    has_nan = torch.isnan(x)
                    has_nan_any_dim = torch.any(has_nan)
                    print(f"x has nan: {has_nan_any_dim.item()}")
                    print("y_pred:", y_pred)
                    '''
                    print("x:", x)
                    print("x_L:", x_L)
                    
                    print("y:", y)
                    print("y_pred_lengths:", y_pred_lengths)
                    print("y_L:", y_L)
                    '''
                    
                    
                y_pred_real = torch.argmax(y_pred,dim=2)

                pred_list += y_pred_real.tolist()
                target_list += y.tolist()

                self.optimizer.zero_grad()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clips)
                self.optimizer.step()

                #train_accuracy += FNT.classification.binary_accuracy(y_pred, y)
                #train_f1 += FNT.classification.binary_f1_score(y_pred, y)
                
                train_loss += loss.detach().cpu()

                if self.scheduler:
                    self.scheduler.step()

                learning_rate = self.scheduler.get_last_lr()[0]
                train_wer += WER(y_pred_real, y)
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

        target_list = []
        pred_list = []

        self.model.eval()
        with torch.no_grad():
            with tqdm(total=len(data), desc=f"EPOCH - {epoch} ") as pbar:
                for step, (x, x_L, y, y_L) in enumerate(data):
                    validation_count += 1

                    if isinstance(x, tuple) or isinstance(x, list):
                        x = [x_.to(device) for x_ in x]
                    else:
                        x = x.to(device)
                
                    if isinstance(y, tuple) or isinstance(y, list):
                        y = [y_.to(device) for y_ in y]
                    else:
                        y = y.to(device)

                    if isinstance(x_L, tuple) or isinstance(x_L, list):
                        x_L = [x_L_.to(device) for x_L_ in x_L]
                    else:
                        x_L = x_L.to(device)

                    if isinstance(y_L, tuple) or isinstance(y_L, list):
                        y_L = [y_L_.to(device) for y_L_ in y_L]
                    else:
                        y_L = y_L.to(device)
                    
                    y_pred, y_pred_lengths = self.model(x,x_L)

       
                    loss = self.loss_function(y_pred.transpose(0,1), y, y_pred_lengths, y_L)
                    y_pred_real = torch.argmax(y_pred,dim=2)

                    pred_list += y_pred_real.tolist()
                    target_list += y.tolist()

                    validation_loss += loss.detach().cpu()
                    #validation_accuracy += FNT.classification.binary_accuracy(y_pred, y)
                    #validation_f1 += FNT.classification.binary_f1_score(y_pred, y)
                    validation_wer += WER(y_pred_real, y)

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

        self.model.zero_grad()
        self.optimizer.zero_grad()

        for epoch in range(self.config.n_epochs):
            train_wer, train_loss, learning_rate = self.train_step(train_dataloader, epoch, device)
            valid_wer, valid_loss = self.validation_step(test_dataloader, epoch, device)

            if valid_wer >= best_score:
                best_model = deepcopy(self.model.state_dict())
                print(f"SAVE! Epoch: {epoch + 1}/{self.config.n_epochs}")
                best_score = valid_wer

                # save model
                if not os.path.exists(self.config.model_dir):
                    os.makedirs(self.config.model_dir)


                model_name = "epoch{}-loss{:.4f}-wer{:.4f}.pt".format(epoch, valid_loss, valid_wer)
                model_path = os.path.join(self.config.model_dir, model_name)
                torch.save({
                    'model': self.model.state_dict(),
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


        
        self.model.load_state_dict(best_model)
