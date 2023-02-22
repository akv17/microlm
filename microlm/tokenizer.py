import numpy as np


class Tokenizer:

    def __init__(self, seq_len):
        self.seq_len = seq_len
        # increment real seq_len to make sure that last token gets assigned real next target via left shift.
        self.seq_len_effective = self.seq_len + 1
        self.i2c = None
        self.c2i = None

    @property
    def num_chars(self):
        return len(self.c2i)

    def encode(self, text):
        ids = [self.c2i[c] for c in text]
        ids = ids[:self.seq_len_effective]
        mask = [0] * len(ids)
        target = np.roll(ids, -1).tolist()
        ids = ids[:self.seq_len]
        mask = mask[:self.seq_len]
        target = target[:self.seq_len]
        enc = {'ids': ids, 'mask': mask, 'target': target}
        return enc

    def decode(self, ids):
        chars = [self.i2c[i] for i in ids]
        text = ''.join(chars)
        return text

    def train(self, text):
        chars = sorted(set(text))
        c2i = {c: i for i, c in enumerate(chars)}
        i2c = {v: k for k, v in c2i.items()}
        self.c2i = c2i
        self.i2c = i2c