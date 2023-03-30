import sys
import numpy as np
from typing import Tuple

import torch
from torch import Tensor, nn
from torchtext.vocab import vocab

from data import (generate_square_subsequent_mask, tokenizer, load_file)
from train import create_model, create_vocab, load_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define special symbols and indices
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3

MAX_SEQUENCE_LENGTH = 29

model = create_model()
model = load_model(model, "./data/test_model.pt")
source_vocab, target_vocab = create_vocab()

def load_file(file_path: str) -> Tuple[Tuple[str], Tuple[str]]:
    """ A helper functions that loads the file into a tuple of strings

    :param file_path: path to the data file
    :return factors: (LHS) inputs to the model
            expansions: (RHS) group truth
    """
    data = open(file_path, "r").readlines()
    factors, expansions = zip(*[line.strip().split("=") for line in data])
    return factors, expansions

def score(true_expansion: str, pred_expansion: str) -> int:
    """ the scoring function - this is how the model will be evaluated

    :param true_expansion: group truth string
    :param pred_expansion: predicted string
    :return:
    """
    return int(true_expansion == pred_expansion)


def greedy_decode(model: nn.Module, src: Tensor, src_mask: Tensor,
                  max_len: int, start_symbol: str):
    """ Greedy search
    :param model: model
    :param src: source
    :param src_mask: source mask
    :param max_len: maximum length
    :param start_symbol: start of sequence
    :return expanded expression in a Tensor format
    """
    model.eval()
    src = src.to(device)
    src_mask = src_mask.to(device)

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)
    for i in range(max_len-1):
        memory = memory.to(device)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0))
                    .type(torch.bool)).to(device)
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == EOS_IDX:
            break
    return ys[1:]

def expand_expression(factor: str, source_vocab: Tensor,
                        target_vocab: Tensor, model: torch.nn.Module, device):
    """Expand Expression
    :param factor: factor to be expanded
    :param source_vocab: factor vocab
    :param target_vocab: expansion vocab
    :param model: model
    :return expanded expression
    """
    model.eval()
    if isinstance(factor, (str, list)):
      tokens = [token for token in tokenizer(factor)]
      tokens = tokens
      src_indexes = [source_vocab.vocab.get_stoi()[token] for token in tokens]
      src_tensor = torch.LongTensor(src_indexes).view(-1, 1).to(device)
    else:
      src_tensor = factor.view(-1, 1).to(device)

    num_tokens = src_tensor.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    tgt_tokens = greedy_decode(
        model,  src_tensor, src_mask, max_len=num_tokens + 5, start_symbol=BOS_IDX).flatten()
    return "".join([target_vocab.vocab.get_itos()[i] for i in tgt_tokens if i != EOS_IDX])

# --------- START OF IMPLEMENT THIS --------- #
def predict(factors: str):
    """Predict the expansion of factor"""

    return expand_expression(factors, source_vocab, target_vocab, model, device)


# --------- END OF IMPLEMENT THIS --------- #

def main(filepath: str):
    factors, expansions = load_file(filepath)
    pred = [predict(f) for f in factors]
    scores = [score(te, pe) for te, pe in zip(expansions, pred)]
    print(np.mean(scores))


if __name__ == "__main__":
    main("test.txt" if "-t" in sys.argv else "train.txt")