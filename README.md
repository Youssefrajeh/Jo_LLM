# Jo

A small GPT-style language model implemented from scratch in PyTorch, trained on the works of Shakespeare, and served through a local web interface that accepts text, text files, and images.

| | |
|---|---|
| Parameters | 10.8 million |
| Architecture | Decoder-only Transformer (6 layers, 6 heads, 384-dim embeddings) |
| Training data | Tiny Shakespeare (1.1 MB) |
| Training time | ~12 minutes on an NVIDIA RTX 4050 Laptop GPU |
| Pretrained weights | None (trained from random initialization) |

## Overview

Every large language model is trained on the same objective: given a sequence of text, predict the next token. Jo implements that objective end to end, with the same core architecture as GPT-2 and GPT-3 (token and position embeddings, stacked self-attention and MLP blocks, and a next-token output layer), at a scale small enough to train on a laptop and to read in full.

Sample output (prompt: `JULIET:\nO Romeo,`):

```
JULIET:
O Romeo, death! what art thou of man?
Ah, and how may she be not for a shoulder there
To be obey'd with us, I will not speak to thee,
Ere thou shalt do say thou be at great to change.
```

The model reproduces the structure of the source text (speaker names, verse line breaks, archaic vocabulary) without any understanding of its meaning.

## Architecture

```mermaid
flowchart LR
    A[Input text] --> B[Character tokenizer]
    B --> C[Token + position embeddings]
    C --> D[Transformer block x6<br/>self-attention + MLP]
    D --> E[LayerNorm + linear head]
    E --> F[Next-character probabilities]
    F -. sample, append, repeat .-> A
```

| | Jo | GPT-2 small | GPT-3 |
|---|---|---|---|
| Layers | 6 | 12 | 96 |
| Attention heads | 6 | 12 | 96 |
| Embedding size | 384 | 768 | 12,288 |
| Context window | 256 characters | 1,024 tokens | 2,048 tokens |
| Parameters | 10.8M | 124M | 175B |

Implementation details:

- Causal multi-head self-attention (PyTorch fused scaled dot-product attention)
- Pre-LayerNorm residual blocks with GELU MLPs
- Weight tying between the input embedding and output layer
- AdamW with decoupled weight decay, linear warmup and cosine learning-rate decay
- bfloat16 mixed precision and gradient clipping
- Checkpointing on best validation loss

## Getting Started

```bash
git clone https://github.com/Youssefrajeh/Jo_LLM.git
cd Jo_LLM
pip install -r requirements.txt
```

For GPU support, install the CUDA build of PyTorch from [pytorch.org](https://pytorch.org). A trained checkpoint is included in `out/`, so no training is required to run the model.

### Web interface

```bash
python app.py
```

Then open http://127.0.0.1:5000. The interface supports:

- Text prompts, with the generated output streamed as it is produced
- Text file uploads, used as the prompt
- Image uploads, converted to a caption by a vision model before being passed to Jo
- Adjustable output length, temperature, and top-k sampling
- An inspection panel showing the exact prompt the model received

The image captioning model (~1 GB) is downloaded from Hugging Face on first use.

### Command line

```bash
python generate.py --prompt "ROMEO:" --tokens 500
python generate.py --prompt "HAMLET:" --temperature 0.5 --top_k 20
```

### Training

```bash
python train.py                                    # default: Tiny Shakespeare, 5,000 steps
python train.py --data corpus.txt --out_dir out_corpus
python generate.py --out_dir out_corpus
```

All hyperparameters (model size, context length, batch size, learning rate, dropout) are configurable via command-line arguments. See `python train.py --help`.

## Results

| Step | Train loss | Validation loss |
|---:|---:|---:|
| 0 | 4.30 | 4.30 |
| 500 | 1.53 | 1.71 |
| 1,000 | 1.26 | 1.51 |
| 1,750 | 1.09 | **1.46** (best) |
| 3,000 | 0.84 | 1.54 |
| 5,000 | 0.58 | 1.72 |

The initial loss matches uniform guessing over the 65-character vocabulary (ln 65 ≈ 4.17). Validation loss reached its minimum at step 1,750 and increased afterwards while training loss continued to fall, indicating overfitting on the 1.1 MB dataset. The included checkpoint is the step-1,750 model. The full log is in `train_log.txt`.

## Multimodal Input

Jo is a text-only model. Images are handled by a separate pretrained captioning model, [BLIP](https://arxiv.org/abs/2201.12086) (`Salesforce/blip-image-captioning-base`), whose output is added to the prompt:

```mermaid
flowchart LR
    I[Image] --> V[BLIP captioning model]
    V --> P[Combined prompt]
    T[Text prompt] --> P
    F[Text files] --> P
    P --> J[Jo]
    J --> O[Generated text]
```

This pipeline is the simplest way to extend a text model to images. Production multimodal models instead project image features directly into the language model's embedding space.

## Limitations

Jo models the surface statistics of its training text. It does not answer questions, follow instructions, or reason. The main differences from production language models are:

1. Scale: billions of parameters trained on trillions of tokens
2. Tokenization: subword (BPE) vocabularies rather than characters
3. Architecture refinements: rotary position embeddings, RMSNorm, SwiGLU, grouped-query attention
4. Post-training: supervised instruction tuning and reinforcement learning from human feedback

## Project Structure

```
Jo_LLM/
├── model.py           Transformer implementation
├── tokenizer.py       Character-level tokenizer
├── train.py           Training loop
├── generate.py        Command-line text generation
├── app.py             Flask server (Jo + BLIP, streaming responses)
├── static/index.html  Web interface
├── data/input.txt     Training data
├── out/               Trained checkpoint and vocabulary
├── train_log.txt      Training log
└── EXPLANATION.md     Detailed walkthrough of every component
```

A step-by-step explanation of the tokenizer, attention mechanism, training procedure, and web application is available in [EXPLANATION.md](EXPLANATION.md).

## References

- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762), 2017
- Radford et al., [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf), 2019
- Li et al., [BLIP: Bootstrapping Language-Image Pre-training](https://arxiv.org/abs/2201.12086), 2022
- Karpathy, [nanoGPT](https://github.com/karpathy/nanoGPT)

## Author

Youssef Rajeh, Emerging Technologies, Semester 6
