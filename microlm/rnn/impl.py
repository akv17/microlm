import os
import json
import logging
from pathlib import Path

import torch


class RnnLanguageModel:

    @classmethod
    def load(cls, path, device='cpu'):
        path = Path(path)
        tokenizer = Tokenizer.load(path)
        model = path.joinpath('model.pt')
        weights = torch.load(model, map_location='cpu')
        model = LSTMModel(vocab_size=tokenizer.vocab_size)
        model.load_state_dict(weights)
        model.eval()
        gen = Generator(tokenizer=tokenizer, model=model)
        ob = cls(device=device, generator=gen)
        return ob

    def __init__(self, device, generator=None):
        self.device = device
        self.generator = generator

    def train(self, path, texts, batch_size=8, epochs=100, seq_len=None, eval_every=10):
        tokenizer = Tokenizer(seq_len=seq_len)
        trainer = Trainer(
            path=path,
            tokenizer=tokenizer,
            device=self.device,
            batch_size=batch_size,
            epochs=epochs,
        )
        trainer.run(texts)

    def generate(self):
        res = self.generator.run()
        return res


class Generator:
    
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model
    
    def run(self):
        generated = [self.tokenizer.start]
        steps = 100
        stop = False
        temp = 1.4
        while not stop:
            steps -= 1
            ctx = generated[-self.tokenizer.seq_len:]
            ids = [self.tokenizer._c2i[ch] for ch in ctx]
            with torch.no_grad():
                ids = torch.tensor(ids)
                logits = self.model(ids)[-1].ravel()
                logits /= temp
                dist = torch.softmax(logits, 0)
                pred = torch.multinomial(dist, num_samples=1).item()
            ch = self.tokenizer._i2c[pred]
            generated.append(ch)
            stop = ch == self.tokenizer.end or steps == 0
        res = ''.join(generated)
        return res


class Tokenizer:

    @classmethod
    def load(cls, path):
        fp = path.joinpath('tokenizer.json')
        with open(fp, 'r') as f:
            data = json.load(f)
        ob = cls()
        ob.seq_len = data['seq_len']
        ob._seq_len_real = ob.seq_len - 2
        ob._c2i = data['c2i']
        ob._i2c = {v: k for k, v in ob._c2i.items()}
        return ob

    def __init__(self, seq_len=None):
        self.seq_len = seq_len
        self.pad = '<P>'
        self.start = '<S>'
        self.end = '<E>'
        self.unk = '<U>'
        self._c2i = None
        self._i2c = None
        self._seq_len_real = None

    @property
    def vocab_size(self):
        return len(self._c2i)
    
    @property
    def start_id(self):
        return self._c2i[self.start]
    
    @property
    def end_id(self):
        return self._c2i[self.end]

    def train(self, texts):
        chars = sorted(set(''.join(texts)))
        chars = [self.pad, self.start, self.end, self.unk] + chars
        self._c2i = {c: i for i, c in enumerate(chars)}
        self._i2c = {v: k for k, v in self._c2i.items()}
        if self.seq_len is None:
            self.seq_len = max([len(t) for t in texts], default=2)
        self._seq_len_real = self.seq_len - 2
    
    def encode(self, text):
        unk_id = self._c2i[self.unk]
        ids = [self._c2i.get(c, unk_id) for c in text]
        ids = ids[:self._seq_len_real]
        ids = [self._c2i[self.start]] + ids + [self._c2i[self.end]]
        loss_mask = [True] * len(ids)
        loss_mask[-1] = False
        pad_size = self.seq_len - len(ids)
        pad_id = self._c2i[self.pad]
        ids += [pad_id] * pad_size
        loss_mask += [False] * pad_size
        enc = {'ids': ids, 'loss_mask': loss_mask}
        return enc

    def decode(self, ids):
        text = [self._i2c.get(i, self.unk) for i in ids]
        text = ''.join(text)
        return text

    def save(self, path):
        fp = path.joinpath('tokenizer.json')
        data = {'seq_len': self.seq_len, 'c2i': self._c2i}
        with open(fp, 'w') as f:
            json.dump(data, f, ensure_ascii=False)


class Trainer:
    
    def __init__(
        self,
        path,
        tokenizer,
        batch_size,
        epochs,
        device,
        workers=None,
    ):
        self.path = Path(path)
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device
        self.workers = workers

        self.logger = None
        self.model = None
        self.loss_fn = None
        self.optimizer = None
        self.dataset = None

    def run(self, texts):
        self._setup_util()
        self.logger.info('Training tokenizer...')
        self.tokenizer.train(texts)
        self.tokenizer.save(self.path)
        self.dataset = Dataset(texts=texts, tokenizer=self.tokenizer)
        if os.getenv('EXPLAIN') == '1':
            self.dataset.explain()
            exit()
        dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.workers or 0
        )
        self.logger.info('Training model...')
        self._setup_model()
        losses = []
        step = 0
        for epoch in range(self.epochs):
            epoch += 1
            for batch in dataloader:
                step += 1
                batch = {k: v.to(self.device) for k, v in batch.items()}
                targets = batch.pop('targets')
                self.optimizer.zero_grad()
                outputs = self.model(**batch)
                outputs = outputs.view(-1, outputs.shape[-1])
                targets = targets.view(-1).long()
                loss = self.loss_fn(outputs, targets)
                loss.backward()
                self.optimizer.step()
                if step % 100 == 0:
                    loss_val = loss.item()
                    losses.append(loss_val)
                    loss_val = torch.tensor(losses).mean().item()
                    msg = f'epoch: {epoch}\tstep: {step}\tloss: {loss:.5f}'
                    self.logger.info(msg)
                    self.model.eval()
                    fp = self.path.joinpath('model.pt')
                    torch.save(self.model.state_dict(), fp)
                    self.model.train()

    
    def _setup_util(self):
        self.path.mkdir(exist_ok=True)
        self.logger = logging.getLogger('Trainer')
        self.logger.setLevel('INFO')
        logging.basicConfig()

    def _setup_model(self):
        self.model = LSTMModel(vocab_size=self.tokenizer.vocab_size)
        self.model.to(self.device)
        self.model.train()
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)


class Dataset(torch.utils.data.Dataset):

    def __init__(self, texts, tokenizer):
        self.texts = texts
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, ix):
        enc = self._encode(ix)
        return enc

    def explain(self, ix=None):
        ix = ix or torch.randint(0, len(self), size=(1,)).item()
        self._explain(ix)
    
    def _encode(self, ix):
        text = self.texts[ix]
        enc = self.tokenizer.encode(text)
        ids = enc['ids']
        ids = torch.tensor(enc['ids'], dtype=torch.int32)
        mask = torch.tensor(enc['loss_mask'])
        targets = torch.roll(ids, -1)
        targets[~mask] = -100
        enc = {'ids': ids, 'targets': targets}
        return enc

    def _explain(self, ix):
        enc = self._encode(ix)
        ids_t = enc['ids']
        targets_t = enc['targets']
        ids = ids_t.tolist()
        targets = targets_t.tolist()
        print(f'ids: {ids_t.shape}')
        print(f'targets: {targets_t.shape}')
        text = self.tokenizer.decode(ids)
        print(f'text: {text}')
        for i_id, t_id in zip(ids, targets):
            if t_id == -100:
                break
            i_ch = self.tokenizer.decode([i_id])
            t_ch = self.tokenizer.decode([t_id])
            print(f'\t{i_ch} -> {t_ch}')
        print()
        print(f'ids: {ids_t}')
        print(f'targets: {targets_t}')


class LSTMModel(torch.nn.Module):

    def __init__(self, vocab_size, embed_size=64, hidden_size=256, pad_id=0):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.pad_id = pad_id

        self.embed = torch.nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.embed_size,
            padding_idx=self.pad_id
        )
        self.lstm = torch.nn.LSTM(
            input_size=self.embed_size,
            hidden_size=self.hidden_size,
            bidirectional=False,
            batch_first=True
        )
        self.head = torch.nn.Linear(self.hidden_size, self.vocab_size)
    
    def forward(self, ids):
        embed = self.embed(ids)
        hidden, _ = self.lstm(embed)
        logits = self.head(hidden)
        return logits


class TransformerModel(torch.nn.Module):

    def __init__(self, vocab_size, embed_size=64, hidden_size=64, pad_id=0):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.pad_id = pad_id

        self.embed = torch.nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.embed_size,
            padding_idx=self.pad_id
        )
        self.transformer = torch.nn.Transformer(
            d_model=64,
            nhead=2,
            num_encoder_layers=2,
            num_decoder_layers=2,
            dim_feedforward=128,
            dropout=0.0,
            batch_first=True
        )
        self.head = torch.nn.Linear(self.hidden_size, self.vocab_size)
    
    def forward(self, ids):
        embed = self.embed(ids)
        hidden = self.transformer(embed, embed)
        logits = self.head(hidden)
        return logits
