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
            for step, (x,z,y) in enumerate(data):

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

                if isinstance(z, tuple) or isinstance(z, list):
                    z = [z_.to(device) for z_ in z]
                else:
                    z = z.to(device)

                
                y_pred, y_pred_lengths = self.model(x,z)
                y_length = torch.ones_like(y.squeeze(1))
                
                loss = self.loss_function(y_pred.transpose(0,1), y, y_pred_lengths, y_length)
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
                for step, (x,z,y) in enumerate(data):
                    validation_count += 1

                    if isinstance(x, tuple) or isinstance(x, list):
                        x = [x_.to(device) for x_ in x]
                    else:
                        x = x.to(device)
                    
                    if isinstance(y, tuple) or isinstance(y, list):
                        y = [y_.to(device) for y_ in y]
                    else:
                        y = y.to(device)

                    if isinstance(z, tuple) or isinstance(z, list):
                        z = [z_.to(device) for z_ in z]
                    else:
                        z = z.to(device)
                    
                    y_pred, y_pred_lengths = self.model(x,z)
                    y_length = torch.ones_like(y.squeeze(1))

                    loss = self.loss_function(y_pred.transpose(0,1), y, y_pred_lengths, y_length)
                    y_pred_real = torch.argmax(y_pred,dim=2)

                    pred_list += y_pred_real.tolist()
                    target_list += y.tolist()

                    validation_loss += loss.detach().cpu()
                    #validation_accuracy += FNT.classification.binary_accuracy(y_pred, y)
                    #validation_f1 += FNT.classification.binary_f1_score(y_pred, y)
                    validation_wer += WER(y_pred_real, y)

                    pbar.update(1)
                    pbar.set_postfix_str(
                        f"valid WER: {validation_wer/validation_count:.2f} Loss: {loss.item():.3f} ({validation_loss / validation_count:.3f})"
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

            print("END")
        
        self.model.load_state_dict(best_model)
