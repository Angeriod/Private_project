import random
import os
import argparse
from datetime import datetime
import warnings
import logging

import numpy as np
import glob
import wandb
import transformers
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

from model import Conformer_L
import dataset
import train


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def define_args():
    p = argparse.ArgumentParser()
    p.add_argument('--wan', type=int, default=1)
    p.add_argument('--name', type=str, default=datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    p.add_argument('--data_dir', type=str, default='./data/')
    p.add_argument('--train_audio_dir', type=str, default='LibriSpeech/train-clean-100/')
    p.add_argument('--train_corpus_dir', type=str, default='librispeech_alignments/train-clean-100/')
    p.add_argument('--test_audio_dir', type=str, default='LibriSpeech/test-clean-100/')
    p.add_argument('--test_corpus_dir', type=str, default='librispeech_alignments/test-clean-100/')
    p.add_argument('--model_dir', type=str, default=f"./save_model/model-{datetime.now().strftime('%Y-%m-%d-%H-%M')}", help="save model path")
    p.add_argument('--gpu_id', type=int, default=0 if torch.cuda.is_available() else -1)

    p.add_argument('--lr', type=float, default=2e-5)
    p.add_argument('--betas', type=float, default=(0.9,0.98))
    p.add_argument('--L2', type=float, default=1e-6)
    p.add_argument('--p', type=float, default=0.01, help="Dropout rate")
    p.add_argument('--clips', type=bool, default=0.6, help="clip grad norm")
    p.add_argument('--batch_size', type=int, default=2)
    p.add_argument('--n_epochs', type=int, default=20)
    p.add_argument('--num_classes', type=int, default=73)
    p.add_argument('--max_seq_len', type=float, default=300)
    p.add_argument('--drop_out', type=float, default=0.1)

    p.add_argument('--valid', type=bool, default=True)
    p.add_argument('--parallel', type=bool, default=False)
    p.add_argument('--seed', type=int, default=-1, help="set seed num")
    p.add_argument('--num_workers', type=float, default=0)

    #mel spectrogram
    p.add_argument('--n_mels', type=int, default=80)
    p.add_argument('--sample_rate', type=int, default=16000)

    #conformer
    p.add_argument('--num_heads', type=int, default=4)
    p.add_argument('--num_layers', type=int, default=16)
    p.add_argument('--depthwise_conv_kernel_size', type=int, default=31)
    p.add_argument('--ffn_dim', type=int, default=256)
    c = p.parse_args()
    return c


def main(config):
    if config.wan >= 1:
        wandb.init(project="Librispeech_conformer_L", entity="2469love", name=config.name)
        wandb.config = {
            "learning_rate": config.lr,
            "epochs": config.n_epochs,
            "batch_size": config.batch_size,
            "drop_out": config.drop_out
        }

    device = torch.device('cpu') if config.gpu_id < 0 else torch.device(f'cuda:{config.gpu_id}')
    # logging.info(f"Working GPU: {device}")
    print(f"Working GPU: {device}")

    model = Conformer_L(config=config).to(device)
    data_train = dataset.CustomDataset(config=config,train=True),
    data_test = dataset.CustomDataset(config=config,train=False),
    
    # data_train = dataset.IterableDataset(data=)
    # data_test = dataset.IterableDataset(data=)
    # 모델 병렬 처리
    if config.parallel:
        model = DistributedDataParallel(model, device_ids=config.gpu_list)
        data_train_sampler = DistributedSampler(data_train)
        data_test_sampler = DistributedSampler(data_test)
        train_dataloader = DataLoader(
            data_train, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, 
            pin_memory=True, sampler=data_train_sampler,collate_fn=train_dataloader.collate_fn)
        test_dataloader = DataLoader(
            data_test, batch_size=config.batch_size,shuffle=False, num_workers=config.num_workers, 
            pin_memory=True, sampler=data_test_sampler,collate_fn=test_dataloader.collate_fn)
       
        # train_dataloader = DataLoader(data_train, batch_size=config.batch_size, num_workers=2, worker_init_fn=dataset.worker_init_fn, pin_memory=True, sampler=data_test_sampler)
        # test_dataloader = DataLoader(data_test, batch_size=config.batch_size, num_workers=2, worker_init_fn=dataset.worker_init_fn, pin_memory=True, sampler=data_test_sampler)

    else:
        train_dataloader = DataLoader(data_train[0], batch_size=config.batch_size,shuffle=True,num_workers=config.num_workers,collate_fn=data_train[0].collate_fn)
        test_dataloader = DataLoader(data_test[0], batch_size=config.batch_size,shuffle=False,num_workers=config.num_workers,collate_fn=data_test[0].collate_fn)
        # train_dataloader = DataLoader(data_train, batch_size=config.batch_size, num_workers=2, worker_init_fn=dataset.worker_init_fn)
        # test_dataloader = DataLoader(data_test, batch_size=config.batch_size, num_workers=2, worker_init_fn=dataset.worker_init_fn)
      
    # scheduler setting
    # 한 에포크 마다 learning rate가 변함
    total = config.n_epochs * len(train_dataloader)
    warmup_rate = 0.1
    optimizer = optim.Adam(params=model.parameters(), lr=config.lr,betas=config.betas, weight_decay= config.L2 )
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer, int(total * warmup_rate), total)
    loss_function = nn.CTCLoss(zero_infinity=True) #ignore_index=-100

    model.zero_grad()
    trainer = train.Trainer(model=model, optimizer=optimizer, loss_function=loss_function, scheduler=scheduler, config=config)
    trainer.fit(train_dataloader=train_dataloader, test_dataloader=test_dataloader, device=device)

    # save model
    # logging.info(f"END")
    print("END")


if __name__ == "__main__":
    config = define_args()
    if config.seed >= 0:
        seed_everything(config.seed)

    main(config=config)


