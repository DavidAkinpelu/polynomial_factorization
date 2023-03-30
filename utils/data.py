""" Data preprocessing functions"""

from collections import Counter
import torch
from torch import Tensor
from torchtext.vocab import vocab

from typing import Tuple, List

# Define special symbols and indices
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3


def load_file(file_path: str) -> Tuple[Tuple[str], Tuple[str]]:
    """ A helper functions that loads the file into a tuple of strings

    :param file_path: path to the data file
    :return factors: (LHS) inputs to the model
            expansions: (RHS) group truth
    """
    data = open(file_path, "r").readlines()
    factors, expansions = zip(*[line.strip().split("=") for line in data])
    return factors, expansions

def tokenizer(string: str) -> List[str]:
    """Convert strings to list of chars
    :params string: string
    :return List: list of chars"""
    tokens = []
    string = string.lower()
    for char in string:
        tokens.append(char)
    return tokens

def data_preprocess(source: List, target: List, source_vocab: vocab,
                    target_vocab: vocab) -> Tuple[List, List]:
    """preprocess both source and target data """
    data = []
    for (raw_source, raw_target) in zip(source, target):
        source_tensor_ = torch.tensor([source_vocab[token] for token in tokenizer(raw_source)],
                                dtype=torch.long)
        target_tensor_ = torch.tensor([target_vocab[token] for token in tokenizer(raw_target)],
                                dtype=torch.long)
        data.append((source_tensor_, target_tensor_))
    return data

def build_vocab(textdata: str, tokenizer: tokenizer) -> vocab:
    """Build vocab"""
    counter = Counter()
    for char in textdata:
        counter.update(tokenizer(char))
    return vocab(counter, min_freq = 1, specials=['<unk>', '<pad>', '<bos>', '<eos>'])

def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generate square mask"""
    mask = (torch.triu(torch.ones((sz, sz))) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src: Tensor, tgt: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """ create mask and padding"""
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len)).type(torch.bool)

    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask