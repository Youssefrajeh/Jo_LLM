"""
Ask questions about your own notes, privately, with Retrieval-Augmented Generation (RAG).

    put .txt / .md files in private/notes/
    python notes_rag.py ask "When is my dentist appointment?"
    python notes_rag.py index        # optional: indexing also happens automatically

Why not train a model on your notes? Training blends facts into millions of weights:
small models recall them unreliably, you can't delete a fact afterwards, and anyone
who gets the weights can coax the secrets out. RAG keeps your notes as files:

    question ──► embed ──► find the most similar note chunks ──► give them to an LLM
                                                                 with the question ──► answer + sources

1. Indexing: every note is split into ~600-character chunks and each chunk is turned
   into a 384-number vector (an "embedding") by bge-small, a model trained so that
   texts with similar meaning get similar vectors.
2. Retrieval: the question is embedded the same way; cosine similarity (dot product
   of normalised vectors) ranks chunks by relevance.
3. Generation: Qwen2.5-1.5B-Instruct reads the top chunks and answers only from them.

Editing or deleting a note is picked up automatically on the next question.
Everything runs on this laptop; notes and the index stay in private/ (git-ignored).
"""

import argparse
import glob
import os
import threading

import torch

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
CHAT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
NOTES_DIR = os.path.join("private", "notes")
INDEX_PATH = os.path.join("private", "notes_index.pt")
NOTE_EXTS = (".txt", ".md")
CHUNK_CHARS = 600
CHUNK_OVERLAP = 100
# bge models expect this prefix on queries (not on documents)
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

device = "cuda" if torch.cuda.is_available() else "cpu"
_models = {}
_lock = threading.Lock()


# ---------------- embeddings ----------------
def _embedder():
    if "embed" not in _models:
        from transformers import AutoModel, AutoTokenizer
        _models["embed"] = (AutoTokenizer.from_pretrained(EMBED_MODEL),
                            AutoModel.from_pretrained(EMBED_MODEL).to(device).eval())
    return _models["embed"]


@torch.no_grad()
def embed(texts, batch_size=32):
    tok, model = _embedder()
    out = []
    for i in range(0, len(texts), batch_size):
        batch = tok(texts[i:i + batch_size], padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
        cls = model(**batch).last_hidden_state[:, 0]          # bge uses the [CLS] token as the sentence vector
        out.append(torch.nn.functional.normalize(cls, dim=-1).float().cpu())
    return torch.cat(out) if out else torch.empty(0, 384)


# ---------------- indexing ----------------
def split_chunks(text):
    """Split on paragraphs, then pack paragraphs into ~CHUNK_CHARS pieces with a little overlap."""
    paras = [p.strip() for p in text.replace("\r\n", "\n").split("\n\n") if p.strip()]
    chunks, cur = [], ""
    for p in paras:
        while len(p) > CHUNK_CHARS:  # a single very long paragraph
            chunks.append(p[:CHUNK_CHARS])
            p = p[CHUNK_CHARS - CHUNK_OVERLAP:]
        if cur and len(cur) + len(p) + 2 > CHUNK_CHARS:
            chunks.append(cur)
            cur = cur[-CHUNK_OVERLAP:] + "\n\n" + p if len(cur) > CHUNK_OVERLAP else p
        else:
            cur = f"{cur}\n\n{p}" if cur else p
    if cur:
        chunks.append(cur)
    return chunks


def note_files(notes_dir):
    files = [f for ext in NOTE_EXTS for f in glob.glob(os.path.join(notes_dir, "**", "*" + ext), recursive=True)]
    return sorted(files)


def fingerprint(notes_dir):
    return {os.path.relpath(f, notes_dir): os.path.getmtime(f) for f in note_files(notes_dir)}


def build_index(notes_dir=NOTES_DIR):
    chunks = []
    for path in note_files(notes_dir):
        with open(path, encoding="utf-8", errors="replace") as f:
            text = f.read()
        rel = os.path.relpath(path, notes_dir)
        chunks += [{"source": rel, "text": c} for c in split_chunks(text)]
    index = {"notes_dir": os.path.abspath(notes_dir), "files": fingerprint(notes_dir), "chunks": chunks,
             "emb": embed([c["text"] for c in chunks]), "embed_model": EMBED_MODEL}
    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
    torch.save(index, INDEX_PATH)
    return index


def load_index(notes_dir=NOTES_DIR):
    """Load the saved index, rebuilding it if any note was added, edited or deleted."""
    if os.path.exists(INDEX_PATH):
        index = torch.load(INDEX_PATH)
        if index["notes_dir"] == os.path.abspath(notes_dir) and index["files"] == fingerprint(notes_dir):
            return index, False
    return build_index(notes_dir), True


def search(question, index, k=4):
    if not index["chunks"]:
        return []
    q = embed([QUERY_PREFIX + question])[0]
    scores = index["emb"] @ q                                   # cosine similarity with every chunk
    top = torch.topk(scores, min(k, len(scores)))
    return [dict(index["chunks"][i], score=round(s, 3)) for s, i in zip(top.values.tolist(), top.indices.tolist())]


# ---------------- answering ----------------
SYSTEM = (
    "You are a private assistant that answers questions about the user using ONLY the notes provided. "
    "Cite the notes you used like [1] or [2]. If the notes don't contain the answer, say you don't find it "
    "in the notes. Do not invent facts. Be concise."
)


def _chat_model():
    if "chat" not in _models:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        _models["chat"] = (AutoTokenizer.from_pretrained(CHAT_MODEL),
                           AutoModelForCausalLM.from_pretrained(CHAT_MODEL, dtype=dtype).to(device).eval())
    return _models["chat"]


def answer_stream(question, hits, max_new_tokens=400):
    """Yields the answer piece by piece."""
    from transformers import TextIteratorStreamer

    tok, model = _chat_model()
    context = "\n\n".join(f"[{i + 1}] (from {h['source']})\n{h['text']}" for i, h in enumerate(hits)) or "(no notes found)"
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"Notes:\n{context}\n\nQuestion: {question}"}]
    inputs = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(device)
    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
    # low temperature: we want faithful answers, not creativity
    kw = dict(**inputs, max_new_tokens=max_new_tokens, streamer=streamer, do_sample=False, pad_token_id=tok.eos_token_id)
    t = threading.Thread(target=lambda: model.generate(**kw))
    t.start()
    yield from streamer
    t.join()


def ask(question, notes_dir=NOTES_DIR, k=4):
    """Retrieve the note chunks most relevant to the question. Returns (hits, index_was_rebuilt, index)."""
    with _lock:
        index, rebuilt = load_index(notes_dir)
    return search(question, index, k), rebuilt, index


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=["index", "ask"])
    p.add_argument("question", nargs="?")
    p.add_argument("--notes_dir", default=NOTES_DIR)
    p.add_argument("-k", type=int, default=4, help="how many note chunks to retrieve")
    args = p.parse_args()

    if not os.path.isdir(args.notes_dir):
        raise SystemExit(f"Create {args.notes_dir} and put .txt/.md notes in it first.")
    if args.cmd == "index":
        idx = build_index(args.notes_dir)
        print(f"Indexed {len(idx['files'])} files into {len(idx['chunks'])} chunks -> {INDEX_PATH}")
    else:
        if not args.question:
            raise SystemExit('Usage: python notes_rag.py ask "your question"')
        hits, rebuilt, idx = ask(args.question, args.notes_dir, args.k)
        if rebuilt:
            print(f"(re-indexed {len(idx['files'])} files, {len(idx['chunks'])} chunks)")
        print("Sources:")
        for i, h in enumerate(hits):
            print(f"  [{i + 1}] {h['source']}  (similarity {h['score']})")
        print()
        for piece in answer_stream(args.question, hits):
            print(piece, end="", flush=True)
        print()
