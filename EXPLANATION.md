# Jo — a small language model built from scratch

Jo is a **GPT-style large language model in miniature**: ~10.8 million parameters,
trained on your laptop GPU (RTX 4050, 6 GB) on the complete works of Shakespeare
(~1.1 MB of text). It uses the same core architecture as GPT-2/3/4 and Llama —
the difference is scale (GPT-3 has 175 **billion** parameters, ~16,000× more).

---

## 1. Files in this folder

| File | What it is |
|---|---|
| `data/input.txt` | Training data: Tiny Shakespeare (1,115,394 characters) |
| `tokenizer.py` | Turns text into numbers and back |
| `model.py` | The neural network (the Transformer) — **the heart of Jo** |
| `train.py` | Teaches Jo by showing it text and correcting its guesses |
| `generate.py` | Makes Jo write new text |
| `out/ckpt.pt` | Jo's learned weights (created by training) |
| `out/tokenizer.json` | Jo's vocabulary (created by training) |
| `train_log.txt` | Loss printed during the training run |
| `app.py` | Web server: runs Jo (and BLIP for images) behind a web page |
| `static/index.html` | The web page for chatting with Jo |
| `README.md` | Project front page for GitHub |
| `requirements.txt` | Python packages needed |
| `code_style.py` | Fine-tunes a pretrained code model on YOUR code with LoRA (section 5c) |
| `notes_rag.py` | Answers questions from YOUR notes with retrieval (RAG) (section 5d) |
| `examples/notes/` | Fictional sample notes for trying out the notes assistant |
| `private/` | Your notes, the notes index and your code adapter. **Git-ignored, never pushed.** |

## 2. How to use it

```bash
python train.py                       # train (~10-15 min on the RTX 4050)
python generate.py                    # write 500 characters
python generate.py --prompt "ROMEO:" --tokens 800 --temperature 0.8
python train.py --data my_file.txt    # train Jo on ANY text file instead
python app.py                         # web page at http://127.0.0.1:5000

python code_style.py train --code_dir "D:\path\to\your\projects"   # learn your coding style
python code_style.py complete --prompt "def load_users(path):" --compare
python notes_rag.py ask "When is my dentist appointment?"         # notes in private/notes/
```

---

## 3. The one idea behind every LLM

> **An LLM is a function that, given some text, predicts the next token.**

That's it. Training = showing it billions of examples of "text → what came next"
and nudging its weights when it guesses wrong. Generating = predict a token,
append it, predict again, repeat. Everything that looks like "intelligence"
in large models emerges from getting extremely good at this one task.

```
input : "To be or not to b"
target: "o be or not to be"   ← the same text shifted by one character
```

At *every* position, Jo guesses the next character. That's why one sequence
of 256 characters gives 256 training examples at once.

---

## 4. What I built, step by step

### Step 1 — Tokenizer (`tokenizer.py`)
Neural networks work on numbers, not letters. We list every distinct character in
the data (65 of them: letters, punctuation, space, newline) and give each an id:

```
'\n'→0, ' '→1, '!'→2, ... 'a'→39, 'b'→40, ...
"hi" → [46, 47]
```

*Real LLMs* use **BPE** (byte-pair encoding): tokens are common chunks like
`"ing"` or `" the"`, with vocabularies of 50k–200k. That makes sequences shorter
so the model can "see" more text at once. Characters are simpler to understand.

### Step 2 — The model (`model.py`)

Data flows like this:

```
token ids (B, T)
   │
   ├─► token embedding    : each id → a vector of 384 numbers ("meaning")
   ├─► position embedding : each position 0..255 → a vector ("where am I")
   │        (added together)
   ▼
 ┌──────────── Block × 6 ────────────┐
 │  LayerNorm → Self-Attention  (+x) │  tokens talk to each other
 │  LayerNorm → MLP             (+x) │  each token thinks on its own
 └───────────────────────────────────┘
   ▼
 LayerNorm → Linear → 65 scores (logits), one per possible next character
   ▼
 softmax → probabilities
```

**Embeddings.** A lookup table: token id → vector of 384 learned numbers.
Similar characters end up with similar vectors. Positions get their own
vectors, because attention by itself has no idea of word order.

**Self-attention — the key invention of the Transformer (2017, "Attention Is All You Need").**
Each token produces three vectors:
- **Query** — "what am I looking for?"
- **Key** — "what do I contain?"
- **Value** — "what do I share if chosen?"

Token *i* compares its query with every earlier token's key (dot product),
turns the scores into weights with softmax, and takes a weighted average of
their values:

```
Attention(Q, K, V) = softmax( Q·Kᵀ / √d  +  causal mask ) · V
```

The **causal mask** blocks each token from looking at future tokens. Otherwise,
when predicting the next character, the model could simply read it — cheating.

**Multi-head** = 6 attention operations in parallel, each with 64 dimensions,
so different heads can learn different patterns (e.g. one tracks "am I inside
a word?", another "who is speaking?").

**MLP (feed-forward).** Two linear layers with a GELU non-linearity in between
(384 → 1536 → 384). Attention *moves* information between tokens; the MLP
*processes* it. Most of an LLM's "knowledge" is thought to be stored in these.

**Residual connections (`x = x + f(x)`)** and **LayerNorm** make deep networks
trainable: gradients can flow straight through, and values stay well-scaled.

**Weight tying.** The input embedding and output layer share the same matrix
(as in GPT-2) — fewer parameters, slightly better results.

**Jo's size:**

| Setting | Jo | GPT-2 small | GPT-3 |
|---|---|---|---|
| Layers | 6 | 12 | 96 |
| Heads | 6 | 12 | 96 |
| Embedding size | 384 | 768 | 12,288 |
| Context length | 256 chars | 1,024 tokens | 2,048 tokens |
| Parameters | **10.8 M** | 124 M | 175 B |

### Step 3 — Training (`train.py`)

Each training step:
1. **Sample a batch** — 64 random 256-character windows from the text.
2. **Forward pass** — Jo predicts the next character at every position.
3. **Loss** — *cross-entropy*: how surprised Jo was by the real next character
   (`-log(probability it gave the correct answer)`).
4. **Backward pass** — PyTorch computes the gradient: how each of the 10.8M
   weights should change to reduce the loss.
5. **Update** — the **AdamW** optimizer nudges every weight a little.

Repeat 5,000 times. Extra techniques used (all standard in real LLM training):

| Technique | Why |
|---|---|
| Train/validation split (90/10) | Val loss on unseen text tells us if Jo is *learning* or just *memorising* |
| Dropout 0.2 | Randomly drops activations so Jo can't over-memorise a 1 MB dataset |
| Weight decay 0.1 | Keeps weights small → better generalisation |
| LR warmup + cosine decay | Start gently, learn fast, then fine-tune slowly at the end |
| Gradient clipping at 1.0 | Prevents rare huge updates from wrecking training |
| bfloat16 mixed precision | ~2× faster and less GPU memory |
| Save best checkpoint | Keeps the weights with the lowest validation loss |

**Reading the loss:** at the start, loss ≈ 4.17 = ln(65), i.e. Jo guesses
uniformly among 65 characters. Lower is better. ~2.5 means Jo has learned
letter frequencies; ~1.5 means it has learned words, spelling and play format.

### Step 4 — Generation (`generate.py`)

```
prompt → Jo → probabilities for next char → sample one → append → repeat
```

- **Temperature** divides the scores before softmax. `< 1` = safer, more
  repetitive; `> 1` = more creative / chaotic.
- **Top-k** only allows the k most likely characters, cutting off nonsense.

---

## 5. Results

Training took about 12 minutes for 5,000 steps on the RTX 4050 (full log in `train_log.txt`):

| Step | Train loss | Val loss | Notes |
|---:|---:|---:|---|
| 0 | 4.30 | 4.30 | random guessing (ln 65 ≈ 4.17) |
| 250 | 1.93 | 2.05 | learned letters, spaces, common words |
| 500 | 1.53 | 1.71 | words and play format |
| 1000 | 1.26 | 1.51 | |
| **1750** | **1.09** | **1.46** | **best validation loss → the saved checkpoint** |
| 3000 | 0.84 | 1.54 | train loss keeps falling, val rises |
| 5000 | 0.58 | 1.72 | clear overfitting |

**The key lesson: overfitting.** After about step 1750, Jo started *memorising* the training
text (train loss kept dropping to 0.58) instead of learning general patterns
(val loss rose again). 1 MB of text is small for a 10.8M-parameter model.
Because `train.py` only saves when val loss improves, the checkpoint in `out/`
is the step-1750 model. Fixes you could try: more data, a smaller model, more
dropout, or fewer steps (`--max_iters 2000`).

Sample output (`python generate.py --prompt "ROMEO:" --seed 7`):

```
ROMEO:
Ay, Juliet, gentlemen! we do my sweet death?

BUCKINGHAM:
The ghost of your son and the glory of the deer,
The corruption of heaven in the world,
Or more like than the manner doth confess;
```

Jo has learned speaker names in capitals followed by a colon, line breaks in verse,
archaic words ("doth", "thee", "thou"), mostly real English words, and even
character names from the right plays. What it lacks is *meaning*.

---

## 5b. The web app (`app.py` + `static/index.html`)

```bash
python app.py     # then open http://127.0.0.1:5000
```

A local web page for talking to Jo. You can type a prompt and/or attach **images**
and **text files** (drag and drop, paste, or 📎).

**How each input reaches Jo:**

```
typed text ─────────────────────────────────┐
text file (.txt .md .py ...) → its contents ─┼─► one prompt ─► Jo ─► streamed reply
image → BLIP vision model → caption ────────┘
```

- **Images:** Jo only understands text, so the server uses **BLIP**
  (`Salesforce/blip-image-captioning-base`), a small *pretrained* image-captioning
  model, to describe the picture in words, e.g. *"a woman sitting on the beach
  with her dog"*. That caption becomes part of Jo's prompt. This
  "vision model → text → LLM" pipeline is the simplest way to make a text-only
  model multimodal. (GPT-4o/Claude instead feed image *embeddings* directly into the
  model.) BLIP downloads (~1 GB) the first time an image is sent.
- **Text files:** decoded as UTF-8; the last 2,000 characters are used (Jo only sees
  its last 256 anyway). Binary files are rejected with a warning.
- **Characters Jo doesn't know** (emoji, accented letters, anything not in its
  65-character vocabulary) are dropped by the tokenizer. The page's
  **"What Jo actually saw"** panel shows the exact prompt after this.

**How streaming works:** `model.generate_stream()` yields each character as soon as
it's sampled. Flask sends them as newline-delimited JSON events
(`caption`, `file`, `prompt`, `token`, `done`), and the page reads the response
stream and appends characters as they arrive, like ChatGPT's typing effect.

**Other details:** a lock keeps one generation on the GPU at a time; the server
reloads the checkpoint automatically if `train.py` saves a newer one; the page
follows the system light/dark theme.

---

## 5c. Your coding style: LoRA fine-tuning (`code_style.py`)

**Goal:** a model that completes code the way *you* write it.

**Why not just train Jo on code?** Jo has 10.8M parameters and has never learned to
program. Trained on your code, it would copy the *look* (indentation, names) but
nothing it wrote would run. Instead we use **transfer learning**: start from
`Qwen2.5-Coder-0.5B`, a 494M-parameter model already trained on billions of lines
of code, and fine-tune it on yours. It keeps its programming ability and picks up
your habits.

**LoRA (Low-Rank Adaptation).** Fine-tuning all 494M weights needs far more than
6 GB of GPU memory (weights, gradients and optimizer state for each). LoRA
freezes the original weights and learns a small correction for each attention
and MLP matrix `W`:

```
W' = W + (alpha / r) · B · A        A: r × d_in,  B: d_out × r,  r = 16
```

Because `r` is tiny compared with `d` (16 vs 896), `A` and `B` together have
far fewer numbers than `W`. Only they are trained: **8.8M parameters, 1.7% of the
model**. The result is a small (tens of MB) "adapter" saved in `private/style_adapter/` that
is applied on top of the unchanged base model.

**What the script does:**
1. Walks your code folder recursively, keeping common source files and skipping
   `node_modules`, `.git`, virtual environments, build output, minified files,
   files over 200 KB, and credential files such as `.env`.
2. Joins all files (each headed by its relative path) and cuts them into
   512-token blocks. 5% are held out for validation.
3. Trains with AdamW (lr 2e-4, warmup + cosine decay), batch 4 × gradient
   accumulation 4 (= effective batch of 16), bfloat16, and **gradient checkpointing**
   (recompute activations during the backward pass instead of storing them, which
   trades speed for memory).
4. Saves the adapter whenever validation loss improves, just like `train.py`.

**Test run** (30 steps on this project's own 7 Python files):
validation loss went from 3.17 to 2.95, and the completions visibly changed style.
For `def get_batch(split):`, the base model wrote a long Google-style docstring;
the fine-tuned model wrote a short inline `#` comment, the way this project does.
Use `--compare` (or the checkbox on the web page) to see both side by side.

**How much code:** aim for 100k+ tokens (roughly 300 KB of source). Default settings
run 2 epochs; on the RTX 4050 each step (16 blocks) takes about 10 seconds.

**Privacy:** a model fine-tuned on your code can reproduce pieces of it, including
anything secret that slipped into the code. That is why the adapter lives in
`private/`, and why credential files are skipped.

---

## 5d. Your notes: Retrieval-Augmented Generation (`notes_rag.py`)

**Goal:** ask questions about yourself ("when is my dentist appointment?") and get
correct answers, privately.

**Why not train a model on your notes?** Training blends facts into millions of
weights. Small models recall facts unreliably and confidently make up the rest;
you can't delete a fact afterwards; and anyone who gets the weights can coax the
secrets back out. **RAG** keeps the facts as plain files and looks them up at
question time:

```
                  ┌──────────── indexing (once, and whenever a note changes) ─────────────┐
 private/notes/*.md, *.txt → split into ~600-char chunks → bge-small → one 384-dim vector each
                  └───────────────────────────────────────────────────────────────────────┘
 question → bge-small → vector → cosine similarity with every chunk → top 4 chunks
        → Qwen2.5-1.5B-Instruct: "answer ONLY from these notes, cite them" → answer + sources
```

**Embeddings.** `BAAI/bge-small-en-v1.5` (33M parameters) is trained so that texts with
similar *meaning* get similar vectors, even with no words in common:
"When do I see the dentist?" lands close to "Dentist appointment: Thursday 9 October".
Vectors are normalised, so similarity is just a dot product (cosine similarity).

**Generation.** `Qwen2.5-1.5B-Instruct` is an instruction-tuned chat model. The system
prompt tells it to use only the provided notes, cite them, and say so when the
answer isn't there. Decoding is greedy (no randomness) for faithful answers.

**Staying up to date.** The index (`private/notes_index.pt`) stores each note's
modification time. If you add, edit or delete a note, the next question rebuilds
the index automatically (a few seconds).

**Test results** on the fictional `examples/notes/`:

| Question | Answer |
|---|---|
| When is my dentist appointment? | Thursday, 9 October 2026 at 10:30 AM. |
| What medicine am I allergic to? | You are allergic to penicillin. |
| Which days do I go to the gym? | Mondays, Wednesdays and Saturdays at 7 pm. |
| What is my bank PIN? | I do not have information about your bank PIN in the given notes. |

The last row is the important one: when the answer isn't in the notes, the model
says so instead of inventing one.

**Privacy.** Everything runs on the laptop; after the one-time model download, no
data leaves it. Notes and the index live in `private/`, which is in `.gitignore`.

**GPU memory.** The code model (~1 GB) and the answer model (~3 GB) don't comfortably
fit in 6 GB alongside everything else, so the web app unloads one when you switch
to the other mode (`free_gpu()` in `app.py`).

---

## 6. What Jo is *not* (and what real LLMs add)

Jo imitates the **style** of Shakespeare but doesn't understand meaning, answer
questions, or follow instructions. ChatGPT/Claude-class models add:

1. **Scale** — billions of parameters, trillions of training tokens, thousands of GPUs for months.
2. **Better tokenizer** — BPE instead of characters.
3. **Architecture tweaks** — RoPE positional encoding, RMSNorm, SwiGLU, grouped-query attention.
4. **Instruction tuning (SFT)** — further training on (question, good answer) pairs so it behaves like an assistant.
5. **RLHF / RLAIF** — reinforcement learning from human or AI feedback to make answers helpful and safe.

But the core — **embeddings → stacked attention + MLP blocks → next-token prediction** — is exactly what you have here.

## 7. Ideas to try next

- Train on your own text: `python train.py --data notes.txt`
- Make Jo bigger: `--n_layer 8 --n_embd 512` (watch GPU memory and val loss)
- Remove dropout (`--dropout 0`) and watch val loss rise while train loss falls = **overfitting**
- Replace the character tokenizer with BPE (the `tiktoken` library)
- Fine-tune a pretrained GPT-2 with Hugging Face `transformers` and compare

## 8. References

- Vaswani et al., *Attention Is All You Need* (2017) — the Transformer paper
- Radford et al., *Language Models are Unsupervised Multitask Learners* (GPT-2, 2019)
- Andrej Karpathy, *Let's build GPT: from scratch, in code, spelled out* (YouTube) and nanoGPT — this project follows the same design
