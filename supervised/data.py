from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import re
import unicodedata
import random
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BOARD_SIZE = 25
TEAM_IDX = {
    'BLUE' : 0,
    'RED' : 1,
    'WHITE' : 2,
    'BLACK' : 3,
}
SPECIAL_TEAM = 4
SOS_TOKEN = '<SOS>'
EOS_TOKEN = '<EOS>'

class Vocab:
    word_to_idx = None
    idx_to_word = None

    def __init__(self, words):
        words_unique = list(set(words))
        if len(words) != len(words_unique):
            print("Warining: found word duplicates!")
        self.idx_to_word = [EOS_TOKEN, SOS_TOKEN] + words_unique
        self.word_to_idx = {}
        for i, word, in enumerate(self.idx_to_word):
            self.word_to_idx[word] = i

    def __len__(self):
        return len(self.idx_to_word)

    # Turn a Unicode string to plain ASCII, thanks to
    # http://stackoverflow.com/a/518232/2809427
    @staticmethod
    def _unicodeToAscii(s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
        )

    @staticmethod
    def _normalize_word(word):
        word = Vocab._unicodeToAscii(word.lower().strip())
        s = re.sub(r"[^a-zA-Z]+", r"", s)

    @staticmethod
    def _load_data(cards_file):
        with open(cards_file, 'r') as f:
            return [Vocab._normalize_word(line) for line in f]

    @staticmethod
    def load_from_file(fname)
        words = Vocab._load_data(fname)
        return Vocab(words)

def generate_board(vocab):
    words = random.sample(vocab.idx_to_word[2:], BOARD_SIZE)  # Sample without replacement
    # Choose a black word
    black_index = random.randrange(0, len(words))

    # Assign cards to teams:
    # BLUE: 0; RED: 1; WHITE: 2; BLACK: 3
    while True:
        team_idx = np.random.randint(low=0, high=3, size=len(words), np.int64)
        team_idx[black_index] = TEAM_IDX['BLACK']
        unique_idx = np.unique(team_idx)
        assert len(unique_idx) <= len(TEAM_IDX)
        # Make sure that all teams are represented. White does not have to be represented.
        if len(unique_idx) == 3 or (len(unique_idx) ==2 and TEAM_IDX['WHITE'] not in unique_idx):
            break
    yield zip(words, team_idx)

def tensors_from_board(vocab, pairs):
    pairs = [(vocab.word_to_idx[word], team) for (word, team) in pairs]
    pairs.append((vocab.word_to_idx['EOS'], SPECIAL_TEAM))
    words_idx, teams_idx = zip(*pairs)
    
    words_tensor = torch.tensor(words_idx, dtype=torch.long, device=device).view(-1, 1)
    teams_tensor = torch.tensor(teams_idx, dtype=torch.long, device=device).view(-1, 1)
    return torch.cat([words_tensor, teams_tensor], 1)
    