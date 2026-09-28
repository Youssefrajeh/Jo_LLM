"""
Generate text from a trained checkpoint.

    python generate.py
    python generate.py --prompt "ROMEO:" --tokens 800 --temperature 0.8 --top_k 40
"""

import argparse
import os

import torch

from model import GPT, GPTConfig
from tokenizer import CharTokenizer

p = argparse.ArgumentParser()
p.add_argument("--out_dir", default="out")
p.add_argument("--prompt", default="\n")
p.add_argument("--tokens", type=int, default=500)
p.add_argument("--temperature", type=float, default=0.8)
p.add_argument("--top_k", type=int, default=40)
p.add_argument("--seed", type=int, default=None)
args = p.parse_args()

if args.seed is not None:
    torch.manual_seed(args.seed)
device = "cuda" if torch.cuda.is_available() else "cpu"

ckpt = torch.load(os.path.join(args.out_dir, "ckpt.pt"), map_location=device)
model = GPT(GPTConfig(**ckpt["config"])).to(device)
model.load_state_dict(ckpt["model"])
model.eval()
tok = CharTokenizer.load(os.path.join(args.out_dir, "tokenizer.json"))

idx = torch.tensor([tok.encode(args.prompt)], dtype=torch.long, device=device)
out = model.generate(idx, args.tokens, temperature=args.temperature, top_k=args.top_k)
print(tok.decode(out[0].tolist()))
