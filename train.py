"""
Train the GPT on a text file.

    python train.py                      # Tiny Shakespeare, default settings
    python train.py --data my_notes.txt  # any plain-text file
    python train.py --max_iters 500      # quick test run
"""

import argparse
import math
import os
import time

import torch

from model import GPT, GPTConfig
from tokenizer import CharTokenizer

p = argparse.ArgumentParser()
p.add_argument("--data", default="data/input.txt")
p.add_argument("--out_dir", default="out")
p.add_argument("--batch_size", type=int, default=64)   # sequences per training step
p.add_argument("--block_size", type=int, default=256)  # context length
p.add_argument("--n_layer", type=int, default=6)
p.add_argument("--n_head", type=int, default=6)
p.add_argument("--n_embd", type=int, default=384)
p.add_argument("--dropout", type=float, default=0.2)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--max_iters", type=int, default=5000)
p.add_argument("--warmup_iters", type=int, default=100)
p.add_argument("--eval_interval", type=int, default=250)
p.add_argument("--eval_iters", type=int, default=100)
args = p.parse_args()

torch.manual_seed(1337)
device = "cuda" if torch.cuda.is_available() else "cpu"
# bfloat16 mixed precision: roughly 2x faster and half the memory on modern NVIDIA GPUs
use_amp = device == "cuda" and torch.cuda.is_bf16_supported()
os.makedirs(args.out_dir, exist_ok=True)

# ---------------- data ----------------
with open(args.data, encoding="utf-8") as f:
    text = f.read()
tok = CharTokenizer.from_text(text)
tok.save(os.path.join(args.out_dir, "tokenizer.json"))

data = torch.tensor(tok.encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]  # hold out 10% to detect overfitting
print(f"{len(text):,} chars | vocab {tok.vocab_size} | train {len(train_data):,} | val {len(val_data):,} tokens")


def get_batch(split):
    """Pick random windows of text. x = tokens [i, i+T), y = tokens [i+1, i+T+1) (the next token at each spot)."""
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - args.block_size - 1, (args.batch_size,))
    x = torch.stack([d[i:i + args.block_size] for i in ix])
    y = torch.stack([d[i + 1:i + args.block_size + 1] for i in ix])
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


# ---------------- model ----------------
cfg = GPTConfig(vocab_size=tok.vocab_size, block_size=args.block_size, n_layer=args.n_layer,
                n_head=args.n_head, n_embd=args.n_embd, dropout=args.dropout)
model = GPT(cfg).to(device)
print(f"Jo: {model.num_params() / 1e6:.2f}M parameters on {device}" + (" (bf16)" if use_amp else ""))

# AdamW: Adam + decoupled weight decay. Don't decay biases/LayerNorm (1-D params).
decay = [p for p in model.parameters() if p.dim() >= 2]
no_decay = [p for p in model.parameters() if p.dim() < 2]
optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1},
                               {"params": no_decay, "weight_decay": 0.0}],
                              lr=args.lr, betas=(0.9, 0.99))


def lr_at(it):
    """Linear warmup, then cosine decay to 10% of max LR. Standard for Transformers."""
    if it < args.warmup_iters:
        return args.lr * (it + 1) / args.warmup_iters
    progress = (it - args.warmup_iters) / max(1, args.max_iters - args.warmup_iters)
    return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(args.eval_iters)
        for k in range(args.eval_iters):
            x, y = get_batch(split)
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_amp):
                _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


# ---------------- training loop ----------------
best_val = float("inf")
t0 = time.time()
for it in range(args.max_iters + 1):
    for g in optimizer.param_groups:
        g["lr"] = lr_at(it)

    if it % args.eval_interval == 0 or it == args.max_iters:
        losses = estimate_loss()
        print(f"step {it:5d} | train loss {losses['train']:.4f} | val loss {losses['val']:.4f} | {time.time() - t0:.0f}s")
        if losses["val"] < best_val:  # keep the checkpoint that generalises best
            best_val = losses["val"]
            torch.save({"model": model.state_dict(), "config": cfg.__dict__, "iter": it, "val_loss": best_val},
                       os.path.join(args.out_dir, "ckpt.pt"))
        if it == args.max_iters:
            break

    x, y = get_batch("train")
    with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_amp):
        _, loss = model(x, y)              # 1. forward: predict next tokens, measure error
    optimizer.zero_grad(set_to_none=True)
    loss.backward()                        # 2. backward: compute gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # avoid rare exploding updates
    optimizer.step()                       # 3. update weights

print(f"done. best val loss {best_val:.4f}. checkpoint: {args.out_dir}/ckpt.pt")
