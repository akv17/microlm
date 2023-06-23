import os

import torch


class NgramLanguageModel:

    def __init__(self, size=1):
        self.size = size
        self._index = None

    def train(self, texts):
        trainer = Trainer(size=self.size, texts=texts)
        index = trainer.run()
        self._index = index

    def generate(self):
        if self._index is None:
            raise Exception('model not trained yet.')
        generator = Generator(size=self.size, index=self._index)
        res = generator.run()
        return res


class Generator:
    
    def __init__(self, size, index):
        self.size = size
        self.index = index
    
    def run(self):
        index = self.index
        generated = [index.start_char]
        stop = False
        while not stop:
            gram = tuple(generated[-self.size:])
            gram_id = index.encode_gram(gram)
            ch_id, prob = index.sample(gram_id)
            ch = index.decode_char(ch_id)
            generated.append(ch)
            stop = ch == index.end_char
            if os.getenv('DEBUG') == '1':
                print(f'{gram} -> {ch} [{prob:.4f}]')
        text = ''.join(c for c in generated if not index.is_spec_char(c))
        return text


class Index:

    def __init__(self, start, end, c2i, g2i, probs):
        self._start = start
        self._end = end
        self._start_id = c2i[self._start]
        self._end_id = c2i[self._end]
        self._c2i = c2i
        self._g2i = g2i
        self._i2c = {v: k for k, v in self._c2i.items()}
        self._i2g = {v: k for k, v in self._g2i.items()}
        self._probs = probs
    
    @property
    def start_char(self):
        return self._start
    
    @property
    def end_char(self):
        return self._end

    def sample(self, i):
        dist = self._probs[i]
        pos = torch.multinomial(dist, num_samples=1).item()
        prob = dist[pos]
        return pos, prob

    def encode_gram(self, value):
        return self._g2i[value]
    
    def encode_char(self, value):
        return self._c2i[value]
    
    def decode_gram(self, value):
        return self._i2g[value]
    
    def decode_char(self, value):
        return self._i2c[value]
    
    def is_spec_char(self, value):
        i = self.encode_char(value)
        return i <= self._end_id


class Trainer:
    
    def __init__(self, size, texts):
        self.size = size
        self.texts = texts
        self._start = '<S>'
        self._end = '<E>'
        self._start_id = 0
        self._end_id = 1
        self._c2i = None
        self._g2i = None
        self._probs = None

    def run(self):
        self._gather_chars()
        self._gather_grams()
        self._compute_maps()
        self._compute_probs()
        index = Index(start=self._start, end=self._end, c2i=self._c2i, g2i=self._g2i, probs=self._probs)
        return index
    
    def _gather_chars(self):
        self._chars = [self._start, self._end] + sorted(set(''.join(self.texts)))
    
    def _gather_grams(self):
        size = self.size
        texts = self.texts
        texts = [[self._start] + list(t) + [self._end] for t in texts]
        self._grams = list(set([tuple(t[max(0, i-size):i]) for t in texts for i in range(1, len(t))]))

    def _compute_maps(self):
        self._c2i = {c: i for i, c in enumerate(self._chars)}
        self._g2i = {g: i for i, g in enumerate(self._grams)}

    def _compute_probs(self):
        num_grams = len(self._g2i)
        num_chars = len(self._c2i)
        counts = torch.zeros((num_grams, num_chars), dtype=torch.int32)
        size = self.size
        g2i = self._g2i
        c2i = self._c2i
        for text in self.texts:
            text = [self._start] + list(text) + [self._end]
            for i in range(1, len(text)):
                g = tuple(text[max(0, i-size):i])
                g_id = g2i[g]
                c = text[i]
                c_id = c2i[c]
                counts[g_id, c_id] += 1
        self._probs = counts.float() / counts.sum(1, keepdim=True)
