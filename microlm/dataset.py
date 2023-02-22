import random

import torch


class Dataset(torch.utils.data.Dataset):
    
    @classmethod
    def load(cls, path, tokenizer):
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()
            text = text.replace('\n', '')
        ob = cls(text=text, tokenizer=tokenizer)
        return ob
    
    def __init__(self, text, tokenizer):
        self.text = text
        self.tokenizer = tokenizer
        self.size = len(text) // self.tokenizer.seq_len_effective
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, ix):
        seq_len = self.tokenizer.seq_len_effective
        start = seq_len * ix
        end = start + seq_len
        text = self.text[start:end]
        enc = self.tokenizer.encode(text)
        enc = {k: torch.as_tensor(v) for k, v in enc.items()}
        return enc

    def split(self, val_size):
        text_train = self.text[:-val_size]
        text_val = self.text[-val_size:]
        train = type(self)(text=text_train, tokenizer=self.tokenizer)
        val = type(self)(text=text_val, tokenizer=self.tokenizer)
        return train, val

    def explain(self, ix='rand'):
        if ix == 'rand':
            ix = random.randint(0, len(self) - 1)
        elif ix == 'last':
            ix = len(self) - 1
        else:
            ix = int(ix)
        enc = self[ix]
        ids = enc['ids'].cpu().numpy()
        target = enc['targets'].cpu().numpy()
        mask = enc['masks'].cpu().numpy()
        print(f'ids:    {ids.shape}')
        print(f'target: {target.shape}')
        print(f'mask:   {mask.shape}')
        inp = self.tokenizer.decode(ids)
        out = self.tokenizer.decode(target)
        print(f'input:  {repr(inp)}')
        print(f'output: {repr(out)}')
        print(f'ids:    {ids[:16]} ...')
        print(f'target: {target[:16]} ...')
        print(f'mask:   {mask[:16]} ...')

