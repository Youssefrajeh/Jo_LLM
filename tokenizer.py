"""
Character-level tokenizer: every distinct character in the training text is one token.

Real LLMs use BPE (byte-pair encoding) with ~50k-200k tokens made of common
character chunks ("ing", " the", ...). Characters keep things simple and the
vocabulary tiny (65 for Shakespeare), at the cost of longer sequences.
"""

import json


class CharTokenizer:
    def __init__(self, chars):
        self.chars = list(chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}  # string -> id
        self.itos = {i: ch for i, ch in enumerate(self.chars)}  # id -> string

    @classmethod
    def from_text(cls, text):
        return cls(sorted(set(text)))

    @property
    def vocab_size(self):
        return len(self.chars)

    def encode(self, s):
        return [self.stoi[c] for c in s if c in self.stoi]  # unknown chars are dropped

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.chars, f)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f))
