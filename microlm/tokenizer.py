import json

import numpy as np


class Tokenizer:

    @classmethod
    def load(cls, path):
        with open(path, 'r') as f:
            data = json.load(f)
        ob = cls(seq_len=data['seq_len'])
        ob.c2i = data['c2i']
        ob.i2c = data['i2c']
        ob.i2c = {int(k): v for k, v in ob.i2c.items()}
        return ob

    def __init__(self, seq_len):
        self.seq_len = seq_len
        # increment real seq_len to make sure that last token gets assigned real next target via left shift.
        self.seq_len_effective = self.seq_len + 1
        self.pad = '[PAD]'
        self.pad_ix = 0
        self.i2c = None
        self.c2i = None

    @property
    def num_chars(self):
        return len(self.c2i)

    def encode_train(self, text):
        ids = [self.c2i[c] for c in text]
        ids = ids[:self.seq_len_effective]
        ids, target = ids[:-1], ids[-1]
        mask = [0] * len(ids)
        pad_size = self.seq_len - len(ids)
        ids += [self.pad_ix] * pad_size
        mask += [1] * pad_size
        enc = {'ids': ids, 'mask': mask, 'target': target}
        return enc

    def encode_generate(self, text):
        ids = [self.c2i[c] for c in text]
        ids = ids[:self.seq_len]
        enc = {'ids': ids}
        return enc

    def decode(self, ids):
        chars = [self.i2c[i] for i in ids if i > self.pad_ix]
        text = ''.join(chars)
        return text

    def train(self, text):
        chars = sorted(set(text))
        c2i = {c: i + 1 for i, c in enumerate(chars)}
        c2i[self.pad] = self.pad_ix
        i2c = {v: k for k, v in c2i.items()}
        self.c2i = c2i
        self.i2c = i2c

    def save(self, path):
        data = {'seq_len': self.seq_len, 'c2i': self.c2i, 'i2c': self.i2c}
        with open(path, 'w') as f:
            json.dump(data, f)
