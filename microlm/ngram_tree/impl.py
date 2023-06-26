import os
import random

import torch


class NgramLanguageModel:

    def __init__(self, size=1):
        self.size = size
        self._tree = None

    def train(self, texts):
        trainer = Trainer(size=self.size, texts=texts)
        self._tree = trainer.run()

    def generate(self):
        res = self._tree.sample()
        return res


class Node:

    def __init__(self, key):
        self.key = key
        self.count = 0
        self._children = {}

    @property
    def children(self):
        return list(self._children.values())

    def __repr__(self):
        return f'Node(key={self.key}, count={self.count}, size={len(self._children)})'

    def __contains__(self, node):
        return node.key in self._children

    def add(self, node):
        if node not in self:
            self._children[node.key] = node
    
    def get(self, key):
        return self._children.get(key)

    def increment(self):
        self.count += 1


class Tree:

    def __init__(self):
        self.root = Node('__root__')
    
    def insert(self, seq):
        parent = self.root
        for el in seq:
            node = parent.get(el)
            if node is None:
                node = Node(el)
                parent.add(node)
            node.increment()
            parent = node
    
    def sample(self):
        generated = []
        parent = self.root
        stop = False
        while not stop:
            children = parent.children
            weights = [n.count for n in children]
            norm = sum(weights)
            weights = [w / norm for w in weights]
            sample = random.choices(children, weights=weights, k=1)[0]
            if os.getenv('DEBUG') == '1':
                print(f'{parent.key} -> {sample.key}')
            generated.append(sample)
            parent = sample
            stop = sample.key == '<E>'
        res = ''.join(n.key for n in generated)
        return res


class Trainer:

    def __init__(self, size, texts):
        self.size = size
        self.texts = texts
        self.start = '<S>'
        self.end = '<E>'
    
    def run(self):
        tree = Tree()
        for text in self.texts:
            text = [self.start] + list(text) + [self.end]
            tree.insert(text)
        return tree