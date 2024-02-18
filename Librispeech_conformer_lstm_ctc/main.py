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
import torchaudio
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset
from torchaudio.models.decoder import download_pretrained_files
from torchaudio.models.decoder import ctc_decoder
import train

from torch.cuda.amp import autocast, GradScaler
from model import Conformer_L, LSTMDecoder, GreedyCharacterDecoder
from utils import model_size, TextTransform, get_audio_transforms, preprocess_example, BatchSampler, TransformerLrScheduler
from subword import Subword


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
    p.add_argument('--wan', type=int, default=-1)
    p.add_argument('--name', type=str, default=datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    p.add_argument('--data_dir', type=str, default='./data/')
    p.add_argument('--train_audio_dir', type=str, default='LibriSpeech/train-clean-100/')
    p.add_argument('--train_corpus_dir', type=str, default='librispeech_alignments/train-clean-100/')
    p.add_argument('--test_audio_dir', type=str, default='LibriSpeech/test-clean-100/')
    p.add_argument('--test_corpus_dir', type=str, default='librispeech_alignments/test-clean-100/')
    p.add_argument('--model_dir', type=str, default=f"./save_model/model-{datetime.now().strftime('%Y-%m-%d-%H-%M')}", help="save model path")
    p.add_argument('--gpu_id', type=int, default=3 if torch.cuda.is_available() else -1)

    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--betas', type=float, default=(0.9,0.98))
    p.add_argument('--L2', type=float, default=1e-6)
    p.add_argument('--p', type=float, default=0.01, help="Dropout rate")
    p.add_argument('--clips', type=bool, default=0.6, help="clip grad norm")
    p.add_argument('--batch_size', type=int, default=6)
    p.add_argument('--n_epochs', type=int, default=20)
    p.add_argument('--num_classes', type=int, default=29)
    p.add_argument('--max_seq_len', type=float, default=300)
    p.add_argument('--drop_out', type=float, default=0.1)

    p.add_argument('--valid', type=bool, default=True)
    p.add_argument('--parallel', type=bool, default=False)
    p.add_argument('--seed', type=int, default=-1, help="set seed num")
    p.add_argument('--num_workers', type=float, default=0)

    #mel spectrogram
    p.add_argument('--n_mels', type=int, default=80)
    p.add_argument('--n_fft', type=int, default=400)
    p.add_argument('--hop_length', type=int, default=160)
    p.add_argument('--sample_rate', type=int, default=16000)
    p.add_argument('--freq_mask_param', type=int, default=27)
    p.add_argument('--time_mask_param', type=int, default=15)
    p.add_argument('--time_mask_param_p', type=float, default=0.05)

    #conformer
    p.add_argument('--num_heads', type=int, default=4)
    p.add_argument('--num_layers', type=int, default=16)
    p.add_argument('--d_model', type=int, default=144)
    p.add_argument('--conv_kernel_size', type=int, default=31)
    p.add_argument('--feed_forward_residual_factor', type=float, default=0.5)
    p.add_argument('--feed_forward_expansion_factor', type=int, default=4)
    p.add_argument('--ffn_dim', type=int, default=256)

    #LSTMdecoder
    p.add_argument('--d_decoder', type=int, default=320)

    #LM
    p.add_argument('--LM_WEIGHT', type=float, default=3.23)
    p.add_argument('--WORD_SCORE', type=float, default=-0.26)
    p.add_argument('--nbest', type=int, default=3)
    p.add_argument('--beam_size', type=int, default=1500)
    p.add_argument('--subword', type=str, default='ASCII')

    p.add_argument('--USE_ONNX', type=bool, default=False)
    c = p.parse_args()
    return c

def main(config):
    if config.wan >= 1:
        wandb.init(project="Conformer-New", entity="2469love", name=config.name)
        wandb.config = {
            "learning_rate": config.lr,
            "epochs": config.n_epochs,
            "batch_size": config.batch_size,
            "drop_out": config.drop_out
        }
    device = torch.device('cpu') if config.gpu_id < 0 else torch.device(f'cuda:{config.gpu_id}')
    # logging.info(f"Working GPU: {device}")
    print(f"Working GPU: {device}")

    encoder = Conformer_L(config=config).to(device)
    decoder = LSTMDecoder(config=config).to(device)
    char_decoder = GreedyCharacterDecoder().eval()

    '''
    files = download_pretrained_files("librispeech-4-gram")
    
  
    beam_search_decoder = ctc_decoder(
        lexicon=files.lexicon,
        tokens=files.tokens,
        lm=files.lm,
        nbest=config.nbest,
        beam_size=config.beam_size,
        lm_weight=config.LM_WEIGHT,
        word_score=config.WORD_SCORE,
    ).to(device).eval()
    '''

    model_size(encoder, 'Encoder')
    model_size(decoder, 'Decoder')
    '''
    USE_ONNX = config.USE_ONNX

    model_vad, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                  model='silero_vad',
                                  force_reload=True,
                                  onnx=USE_ONNX)
    '''
    if not os.path.isdir(config.data_dir):
        os.makedirs(config.data_dir)

    data_train = torchaudio.datasets.LIBRISPEECH(config.data_dir, url="train-clean-100", download=True),
    data_test = torchaudio.datasets.LIBRISPEECH(config.data_dir, url='test-clean', download=True),

    subword_loader = Subword(data_train,data_test)
    if not os.path.exists('SentencePiece_bpe.vocab') or os.path.exists('SentencePiece_ngram.vocab'):
        subword_loader.make_txt()

    sorted_train_ids = [idx for idx, _ in sorted(enumerate(data_train[0]), key=lambda x: x[1][1])]
    sorted_test_ids = [idx for idx, _ in sorted(enumerate(data_test[0]), key=lambda x: x[1][1])]
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
        train_dataloader = DataLoader(
            dataset=data_train[0],
            batch_sampler= BatchSampler(sorted_train_ids, batch_size=config.batch_size),
            num_workers=config.num_workers,
            collate_fn=lambda x: preprocess_example(x, config, subword_loader,'train', config.subword))

        test_dataloader = DataLoader(
            dataset=data_test[0],
            batch_sampler= BatchSampler(sorted_test_ids, batch_size=config.batch_size),
            num_workers=config.num_workers,
            collate_fn=lambda x: preprocess_example(x, config, subword_loader,'test', config.subword ))
        # train_dataloader = DataLoader(data_train, batch_size=config.batch_size, num_workers=2, worker_init_fn=dataset.worker_init_fn)
        # test_dataloader = DataLoader(data_test, batch_size=config.batch_size, num_workers=2, worker_init_fn=dataset.worker_init_fn)

    print(len(train_dataloader))


    # scheduler setting
    # 한 에포크 마다 learning rate가 변함
    optimizer = optim.AdamW(params=list(encoder.parameters()) + list(decoder.parameters()), lr=config.lr, betas=config.betas, eps=1e-05 if False else 1e-09 ,weight_decay= config.L2 )
    scheduler = TransformerLrScheduler(optimizer, config.d_model, 10000)
    loss_function = nn.CTCLoss(blank=28,zero_infinity=True) #ignore_index=-100
    grad_scaler = GradScaler(enabled=False)

    trainer = train.Trainer(encoder=encoder, decoder=decoder, char_decoder=char_decoder, optimizer=optimizer,
                            loss_function=loss_function, scheduler=scheduler, grad_scaler=grad_scaler, config=config, subword=subword_loader)
    trainer.fit(train_dataloader=train_dataloader, test_dataloader=test_dataloader, device=device)

    # save model
    # logging.info(f"END")
    print("END")


if __name__ == "__main__":
    config = define_args()
    if config.seed >= 0:
        seed_everything(config.seed)

    main(config=config)


