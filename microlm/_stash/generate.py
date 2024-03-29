import os

import click
import torch

from .tokenizer import Tokenizer
from .nn import Transformer, LSTM
from .util import create_logger, load_config


class Generator:

    @classmethod
    def from_config(cls, config):
        dst = config['train']['dst']
        device = config['generate']['device']
        tokenizer = os.path.join(dst, 'tokenizer.json')
        tokenizer = Tokenizer.load(tokenizer)
        model = LSTM.from_config(config, tokenizer=tokenizer)
        weights = os.path.join(dst, 'model.pt')
        weights = torch.load(weights, map_location=device)
        model.load_state_dict(weights)
        model.eval()
        model.to(device)
        max_size = config['generate'].get('max_size', 1024)
        temp = config['generate'].get('temperature', 1.0)
        ob = cls(tokenizer=tokenizer, model=model, max_size=max_size, temperature=temp)
        return ob

    def __init__(self, tokenizer, model, max_size=1024, temperature=1.0):
        self.tokenizer = tokenizer
        self.model = model
        self.max_size = max_size
        self.temperature = temperature

    def run(self, prompt):
        buffer = list(prompt)
        seq_len = self.tokenizer.seq_len
        with torch.no_grad():
            while len(buffer) < self.max_size:
                prompt = buffer[-seq_len:] 
                prompt = ''.join(prompt)
                enc = self.tokenizer.encode_generate(prompt)
                ids = enc['ids']
                ids = torch.as_tensor(ids)
                ids = ids.unsqueeze(0)
                mask = [0] * len(ids)
                mask = torch.as_tensor(mask)
                out = self.model(ids, mask=mask)
                logits = out[0]
                logits = logits.cpu()
                logits = logits / self.temperature
                probs = torch.softmax(logits, dim=-1)
                next_ix = torch.multinomial(probs, num_samples=1).item()
                # next_ix = probs.argmax().item()
                pred = self.tokenizer.decode([next_ix])
                buffer.append(pred)
            print(''.join(buffer))


@click.command()
@click.argument('config')
@click.argument('prompt')
def main(config, prompt):
    config = load_config(config)
    generator = Generator.from_config(config)
    generator.run(prompt)
    

if __name__ == '__main__':
    main()
