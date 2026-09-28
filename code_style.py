"""
Teach a small pretrained code model YOUR coding style with LoRA fine-tuning.

    python code_style.py train --code_dir "D:\\path\\to\\your\\projects"
    python code_style.py complete --prompt "def load_users(path):"
    python code_style.py complete --prompt "def load_users(path):" --compare   # also show the base model

Why not train Jo on code? Jo is 10.8M parameters and has never learned to program,
so it would copy the *look* of your code but write nothing that runs. Instead we start
from Qwen2.5-Coder-0.5B, which already knows how to program, and nudge it toward your
habits (naming, structure, comments, libraries you use).

LoRA (Low-Rank Adaptation): instead of updating all 494M weights, freeze them and learn
a small correction for each attention/MLP matrix W:
        W' = W + (alpha / r) * B @ A        A: (r x d_in), B: (d_out x r), r = 16
Only A and B are trained (~9M parameters, <2% of the model), so it fits in 6 GB of
GPU memory and the result is a small (tens of MB) "adapter" file applied on top of the base model.

Everything (dataset and adapter) is written to private/, which git ignores:
a model fine-tuned on your code can reproduce pieces of it.
"""

import argparse
import json
import math
import os
import random
import time

import torch

BASE_MODEL = "Qwen/Qwen2.5-Coder-0.5B"
ADAPTER_DIR = os.path.join("private", "style_adapter")

CODE_EXTS = {".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".c", ".h", ".cpp", ".hpp", ".cs", ".go",
             ".rs", ".php", ".rb", ".kt", ".swift", ".sql", ".html", ".css", ".scss", ".sh", ".ps1", ".dart"}
SKIP_DIRS = {".git", "node_modules", "venv", ".venv", "env", "__pycache__", "build", "dist", "bin", "obj",
             ".next", ".idea", ".vscode", "target", "vendor", "private", "out", ".gradle", "migrations"}
# never feed credentials to the model, even privately
SKIP_FILES = {".env", "secrets.py", "credentials.json", "config.local.js"}
MAX_FILE_BYTES = 200_000  # skip huge generated / minified files


# ---------------- data ----------------
def collect_code(code_dir, exts):
    files = []
    for root, dirs, names in os.walk(code_dir):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")]
        for name in names:
            path = os.path.join(root, name)
            if (os.path.splitext(name)[1].lower() not in exts or name in SKIP_FILES
                    or name.endswith(".min.js") or os.path.getsize(path) > MAX_FILE_BYTES):
                continue
            try:
                with open(path, encoding="utf-8") as f:
                    text = f.read()
            except (UnicodeDecodeError, OSError):
                continue
            if text.strip():
                files.append((os.path.relpath(path, code_dir), text))
    return files


def make_blocks(files, tok, block_size):
    """Concatenate all files (separated by end-of-text) and cut into fixed-length token blocks."""
    ids = []
    for rel, text in files:
        ids += tok(f"# {rel}\n{text}", add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
    n = len(ids) // block_size
    return torch.tensor(ids[: n * block_size]).view(n, block_size), len(ids)


# ---------------- model ----------------
def load_model(adapter=True, for_training=False):
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=dtype).to(device)

    if for_training:
        model.gradient_checkpointing_enable()   # recompute activations in backward: less memory, a bit slower
        model.enable_input_require_grads()
        cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
        model = get_peft_model(model, cfg)
    elif adapter:
        if not os.path.isdir(ADAPTER_DIR):
            raise FileNotFoundError(f"No adapter in {ADAPTER_DIR}. Run `python code_style.py train --code_dir ...` first.")
        model = PeftModel.from_pretrained(model, ADAPTER_DIR)
    model.eval()
    return model, tok, device


# ---------------- train ----------------
def train(args):
    exts = {e if e.startswith(".") else "." + e for e in args.exts.split(",")} if args.exts else CODE_EXTS
    files = collect_code(args.code_dir, exts)
    if not files:
        raise SystemExit(f"No code files found in {args.code_dir}")
    by_ext = {}
    for rel, _ in files:
        ext = os.path.splitext(rel)[1].lower()
        by_ext[ext] = by_ext.get(ext, 0) + 1
    print(f"Found {len(files)} files: " + ", ".join(f"{k} {v}" for k, v in sorted(by_ext.items(), key=lambda x: -x[1])))

    model, tok, device = load_model(for_training=True)
    model.print_trainable_parameters()

    random.seed(0)
    random.shuffle(files)
    blocks, n_tokens = make_blocks(files, tok, args.block_size)
    n_val = max(1, len(blocks) // 20)  # 5% held out to watch for overfitting
    train_b, val_b = blocks[n_val:], blocks[:n_val]
    print(f"{n_tokens:,} tokens -> {len(train_b)} train / {len(val_b)} val blocks of {args.block_size}")
    if len(train_b) < 8:
        print("Warning: very little code. The model will mostly memorise it. Aim for 100k+ tokens.")

    steps_per_epoch = math.ceil(len(train_b) / (args.batch_size * args.grad_accum))
    total_steps = args.max_steps or steps_per_epoch * args.epochs
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 20) * 0.5 * (1 + math.cos(math.pi * min(s, total_steps) / total_steps)))

    @torch.no_grad()
    def val_loss():
        model.eval()
        losses = []
        for i in range(0, len(val_b), args.batch_size):
            x = val_b[i:i + args.batch_size].to(device)
            losses.append(model(input_ids=x, labels=x).loss.item())
        model.train()
        return sum(losses) / len(losses)

    os.makedirs(ADAPTER_DIR, exist_ok=True)
    best = val_loss()
    print(f"step 0/{total_steps} | val loss {best:.4f}  (base model, before fine-tuning)")
    model.train()
    step, t0 = 0, time.time()
    while step < total_steps:
        order = torch.randperm(len(train_b))
        for i in range(0, len(order), args.batch_size * args.grad_accum):
            for j in range(args.grad_accum):  # gradient accumulation: several small batches = one bigger batch
                idx = order[i + j * args.batch_size: i + (j + 1) * args.batch_size]
                if len(idx) == 0:
                    break
                x = train_b[idx].to(device)
                loss = model(input_ids=x, labels=x).loss / args.grad_accum  # labels=x: HF shifts by one internally
                loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % args.eval_every == 0 or step == total_steps:
                v = val_loss()
                mark = ""
                if v < best:
                    best = v
                    model.save_pretrained(ADAPTER_DIR)
                    mark = "  saved"
                print(f"step {step}/{total_steps} | train loss {loss.item() * args.grad_accum:.4f} | val loss {v:.4f} | {time.time() - t0:.0f}s{mark}")
            if step >= total_steps:
                break

    if not os.path.exists(os.path.join(ADAPTER_DIR, "adapter_config.json")):
        model.save_pretrained(ADAPTER_DIR)  # val never improved; keep the final weights anyway
    with open(os.path.join(ADAPTER_DIR, "jo_meta.json"), "w") as f:
        json.dump({"base_model": BASE_MODEL, "files": len(files), "tokens": n_tokens, "best_val_loss": best,
                   "steps": total_steps, "trained_at": time.strftime("%Y-%m-%d %H:%M")}, f, indent=2)
    print(f"done. best val loss {best:.4f}. adapter saved to {ADAPTER_DIR}")


# ---------------- generate ----------------
def complete_stream(model, tok, device, prompt, max_new_tokens=200, temperature=0.4, use_adapter=True):
    """Yields text pieces as they are generated. use_adapter=False runs the original base model."""
    from contextlib import nullcontext
    from threading import Thread

    from transformers import TextIteratorStreamer

    inputs = tok(prompt, return_tensors="pt").to(device)
    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
    kw = dict(**inputs, max_new_tokens=max_new_tokens, streamer=streamer, pad_token_id=tok.eos_token_id,
              do_sample=temperature > 0, repetition_penalty=1.1)
    if temperature > 0:
        kw.update(temperature=temperature, top_p=0.95)

    def run():
        ctx = nullcontext() if use_adapter or not hasattr(model, "disable_adapter") else model.disable_adapter()
        with ctx, torch.no_grad():
            model.generate(**kw)

    t = Thread(target=run)
    t.start()
    yield from streamer
    t.join()


def complete(args):
    model, tok, device = load_model()
    runs = [("your style", True)] + ([("base model", False)] if args.compare else [])
    for label, use in runs:
        print(f"\n===== {label} =====\n{args.prompt}", end="", flush=True)
        for piece in complete_stream(model, tok, device, args.prompt, args.tokens, args.temperature, use):
            print(piece, end="", flush=True)
        print()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train", help="fine-tune on a folder of your code")
    t.add_argument("--code_dir", required=True, help="folder with your projects (searched recursively)")
    t.add_argument("--exts", default=None, help="comma-separated extensions to include, e.g. py,js (default: common languages)")
    t.add_argument("--block_size", type=int, default=512)
    t.add_argument("--batch_size", type=int, default=4)
    t.add_argument("--grad_accum", type=int, default=4)
    t.add_argument("--epochs", type=int, default=2)
    t.add_argument("--max_steps", type=int, default=None)
    t.add_argument("--lr", type=float, default=2e-4)
    t.add_argument("--eval_every", type=int, default=25)
    c = sub.add_parser("complete", help="complete code in your style")
    c.add_argument("--prompt", required=True)
    c.add_argument("--tokens", type=int, default=200)
    c.add_argument("--temperature", type=float, default=0.4)
    c.add_argument("--compare", action="store_true", help="also show what the base model writes")
    args = p.parse_args()
    train(args) if args.cmd == "train" else complete(args)
