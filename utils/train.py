""" Script to train a model using a sequence to sequence transformer"""

from multiprocessing.spawn import get_preparation_data
import time
import math
import argparse
import torch
from torch import nn, Tensor
from torch.optim import Optimizer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from typing import Tuple, List, Optional

from model import Seq2SeqTransformer
from data import (create_mask, tokenizer,
                  load_file, build_vocab, data_preprocess)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define special symbols and indices
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3


def collate_fn(data_batch: List) -> Tuple[Tensor, Tensor]:
    """Pad source and target sequence and add EOS and BOS
    param batch: list of Tensor
    :return sr_batch: padded sequence
    :return tg_batch: padded sequence"""

    sr_batch, tg_batch = [], []
    for (de_item, en_item) in data_batch:
        sr_batch.append(torch.cat([torch.tensor([BOS_IDX]), de_item, torch.tensor([EOS_IDX])], dim=0))
        tg_batch.append(torch.cat([torch.tensor([BOS_IDX]), en_item, torch.tensor([EOS_IDX])], dim=0))
    sr_batch = pad_sequence(sr_batch, padding_value=PAD_IDX)
    tg_batch = pad_sequence(tg_batch, padding_value=PAD_IDX)
    return sr_batch, tg_batch


def train(model: nn.Module, train_iter: DataLoader, optimizer: Optimizer,
          loss_fn: nn.Module) -> float:
    """Train model
    :param model: model
    :param train_iter: training data
    :param optimier: optimizer:
    :param loss_fn: train loss
    :return average loss
    """
    model.train()
    losses = 0

    for src, tgt in train_iter:
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

        optimizer.zero_grad()

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()

        optimizer.step()
        losses += loss.item()

    return losses / len(train_iter)

def evaluate(model: nn.Module, val_iter: DataLoader, loss_fn: nn.Module) -> float:
    """Evaluate model
     :param model: model
    :param train_iter: training data
    :param loss_fn: val loss
    :return average loss"""
    model.eval()
    losses = 0

    for src, tgt in val_iter:
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()

    return losses / len(val_iter)

def epoch_time(start_time, end_time) -> Tuple[int, int]:
    """Calculates difference between start and end time in minutes and seconds"""
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def count_parameters(model: nn.Module) -> float:
    """Count number of trainable parameters in the model
    :param model: model
    :return:"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_epoch(n_epochs: int, model: nn.Module, model_name: str, train_iter: DataLoader,
               val_iter: DataLoader, optimizer: Optimizer, train_loss_fn: nn.Module, val_loss_fn: nn.Module):
    """Training iteration function
    :param n_epochs: number of epochs
    :param model: model
    :param model_name: name of the model
    :param train_iter: training data
    :param val_iter: validation data
    :param optimier: optimizer:
    :param train_loss_fn: train loss
    :param val_loss_fn: validation loss
    :return trained model"""

    best_valid_loss = float('inf')


    for epoch in range(n_epochs):

        start_time = time.time()

        train_loss = train(model, train_iter, optimizer, train_loss_fn)
        valid_loss = evaluate(model, val_iter, val_loss_fn)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), model_name + '.pt')

        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

    return model

def load_model(model: nn.Module, model_name: str):
    """Load model
    :param model: initialize model
    :param model_name: filename of the saved model
    :return model: pytorch model with weight"""

    state_dict = torch.load(model_name, map_location=device)
    model.load_state_dict(state_dict)
    return model.to(device)

def create_model(num_encoder_layers: int=6, num_decoder_layers: int=6,
                emb_size:int = 128,
                n_head: int=8, source_vocab_size: int=32, target_vocab_size: int=32,
                ffn_hid_dim: int=512):
    """ Initialize model
    :param num_encoder_layers: number of encoder layers
    :param num_decoder_layers: number of decoder layers
    :param emb_size: embedding size
    :param n_head: number of heads
    :source_vocab_size: size of source vocab
    :target_vocab_size: size of target vocab
    :params ffn_hid_dim: Feed forwar dimension
    :return model: initilaized model
    """

    model = Seq2SeqTransformer(num_encoder_layers, num_decoder_layers, emb_size,
                n_head, source_vocab_size, target_vocab_size, ffn_hid_dim)

    return model

def create_vocab(filename: Optional[str]=None):
    """Create Vocabulary"""
    if filename is None:
        filename  = "./data/data.txt"
    source, target = load_file(filename)
    train_source, test_source, train_target,test_target = train_test_split(source, target,
                                                                           test_size=0.2, random_state=42)

    source_vocab = build_vocab(train_source, tokenizer)
    target_vocab = build_vocab(train_target, tokenizer)
    return source_vocab, target_vocab



def run(file, model_name, batch_size, num_epochs, num_encoder_layers, num_decoder_layers, emb_size,
         n_head, ffn_hid_dim):

    source, target = load_file(file)

    train_source, test_source, train_target,test_target = train_test_split(source, target,
                                                                           test_size=0.2, random_state=42)
    val_source, test_source, val_target, test_target = train_test_split(test_source, test_target,
                                                                        test_size=0.5, random_state=42)

    source_vocab = build_vocab(train_source, tokenizer)
    target_vocab = build_vocab(train_target, tokenizer)

    train_data = data_preprocess(train_source, train_target, source_vocab, target_vocab)
    val_data = data_preprocess(val_source, val_target, source_vocab, target_vocab)
    test_data = data_preprocess(test_source, test_target, source_vocab, target_vocab)

    SRC_VOCAB_SIZE = len(source_vocab)
    TGT_VOCAB_SIZE = len(target_vocab)



    train_iter = DataLoader(train_data, batch_size=batch_size,
                        shuffle=True, collate_fn=collate_fn)
    val_iter = DataLoader(val_data, batch_size=batch_size,
                        shuffle=True, collate_fn=collate_fn)
    test_iter = DataLoader(test_data, batch_size=batch_size,
                       shuffle=True, collate_fn=collate_fn)

    model = Seq2SeqTransformer(num_encoder_layers, num_decoder_layers, emb_size,
                                n_head, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, ffn_hid_dim)


    model = model.to(device)

    train_loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX, label_smoothing=0.1)

    val_loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

    train_epoch(num_epochs, model, model_name, train_iter, val_iter, optimizer, train_loss_fn, val_loss_fn)

    test_loss = evaluate(model, test_iter, val_loss_fn)

    print(f'| Test Loss: {test_loss:.5f} | Test PPL: {math.exp(test_loss):7.3f} |')

    return


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Transformer testing')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch_size')
    parser.add_argument('--n_epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--file', type=str, default='./data/data.txt', help='Filename containing  the data')
    parser.add_argument('--model_name', type=str, default='transformer.pt', help='Name to save the model with')
    parser.add_argument('--n_enc_layers', type=int, default=6, help='Number of encoder layers')
    parser.add_argument('--n_dec_layers', type=int, default=6, help='Number of decoder layers')
    parser.add_argument('--n_heads', type=int, default=8, help='Number of heads')
    parser.add_argument('--emb_size', type=int, default=128, help='Embedding size')
    parser.add_argument('--ffn_hid_dim', type=int, default=128, help='Feed Forward Hidden dimension')

    args = parser.parse_args()

    file = args.file
    model_name = args.model_name
    batch_size = args.batch_size
    num_epochs = args.n_epochs
    num_encoder_layers = args.n_enc_layers
    num_decoder_layers = args.n_dec_layers
    emb_size = args.emb_size
    n_head = args.n_heads
    ffn_hid_dim = args.ffn_hid_dim

    run(file, model_name, batch_size, num_epochs, num_encoder_layers, num_decoder_layers, emb_size,
         n_head, ffn_hid_dim)