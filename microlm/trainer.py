import os
import json
import time

import torch
import numpy as np


class Trainer:
    
    def __init__(
        self,
        logger,
        dst,
        device,
        dataset_train,
        dataset_val,
        model,
        batch_size=1,
        epochs=1,
        workers=0,
        log_every=1,
        eval_every=1,
    ):
        self.logger = logger
        self.dst = dst
        self.device = device
        self.dataset_train = dataset_train
        self.dataset_val = dataset_val
        self.model = model
        self.batch_size = batch_size
        self.epochs = epochs
        self.workers = workers
        self.log_every = log_every
        self.eval_every = eval_every
        self.state = None
    
    def run(self):
        self.setup()
        step = 0
        timer = time.perf_counter()
        for epoch in range(self.epochs):
            for batch in self.dataloader_train:
                step += 1
                batch = {k: v.to(self.device) for k, v in batch.items()}
                targets = batch.pop('target')
                self.optimizer.zero_grad()
                outputs = self.model(**batch)
                loss = self.loss_fn(outputs, targets)
                loss.backward()
                self.optimizer.step()
                self.state['time'] = time.perf_counter() - timer
                self.state['epoch'] = epoch
                self.state['step'] = step
                self.state['loss'] = loss.item()
                self.on_step_end()
                timer = time.perf_counter()

    def setup(self):
        os.makedirs(self.dst, exist_ok=True)
        self.state = {'losses': []}
        self.dataloader_train = torch.utils.data.DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            num_workers=self.workers,
            shuffle=True
        )
        self.dataloader_val = torch.utils.data.DataLoader(
            self.dataset_val,
            batch_size=self.batch_size,
            num_workers=self.workers,
        )
        self.model.train()
        self.model.to(self.device)
        self.loss_fn = Loss()
        self.optimizer = torch.optim.AdamW(self.model.parameters())

    def on_step_end(self):
        step = self.state['step']
        if step % self.log_every == 0:
            self.state['losses'].append(self.state['loss'])
            loss = np.mean(self.state['losses'])
            ep = self.state['epoch']
            time_ = self.state['time']
            msg = f'epoch: {ep} — step: {step} — loss: {loss:.5f} — time: {time_:.4f}'
            self.logger.info(msg)
        if step % self.eval_every == 0:
            self.model.eval()
            self.eval()
            self.checkpoint()
            self.model.train()

    def eval(self):
        losses = []
        with torch.no_grad():
            for batch in self.dataloader_val:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                targets = batch.pop('target')
                outputs = self.model(**batch)
                loss = self.loss_fn(outputs, targets)
                loss = loss.item()
                losses.append(loss)
        score = float(np.mean(losses))
        self.state['score'] = score
        self.logger.info(f'eval: {score}')

    def checkpoint(self):
        best_score = self.state.get('best_score')
        score = self.state['score']
        if best_score is None or score <= best_score:
            self.state['best_score'] = score
            tok_fp = os.path.join(self.dst, 'tokenizer.json')
            self.dataset_train.tokenizer.save(tok_fp)
            model_fp = os.path.join(self.dst, 'model.pt')            
            torch.save(self.model.state_dict(), model_fp)
            state = self.state.copy()
            state.pop('losses')
            state_fp = os.path.join(self.dst, 'state.json')            
            with open(state_fp, 'w') as f:
                json.dump(state, f)
            self.logger.info(f'checkpoint: {score}')
        model_fp = os.path.join(self.dst, 'model_last.pt')
        torch.save(self.model.state_dict(), model_fp)


class Loss(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.fn = torch.nn.CrossEntropyLoss()
    
    def forward(self, outputs, targets):
        outputs = outputs.view(-1, outputs.size(-1))
        targets = targets.view(-1)
        loss = self.fn(outputs, targets)
        return loss
 