"""
Web app for talking to Jo, plus two private assistants.

    python app.py        then open http://127.0.0.1:5000

Three modes (tabs at the top of the page):
    Shakespeare  /api/generate  Jo, the model trained from scratch (below)
    My code      /api/code      Qwen2.5-Coder fine-tuned on your code with LoRA (code_style.py)
    My notes     /api/ask       answers from private/notes via RAG (notes_rag.py)

Environment variables: JO_OUT_DIR (Jo checkpoint folder), JO_NOTES_DIR (notes folder), JO_PORT.

How a Shakespeare request flows:
    typed text ─────────────────────────────┐
    text files (.txt, .md, ...) → contents ─┼─► one prompt ─► Jo ─► streamed text
    images → BLIP (vision model) → caption ─┘

Jo only understands text, so images are first turned into words by BLIP,
a small pretrained image-captioning model. This "vision model → text → LLM"
pipeline is the simplest way to make a text-only LLM multimodal.
"""

import io
import json
import os
import threading

import torch
from flask import Flask, Response, jsonify, request, send_from_directory, stream_with_context
from PIL import Image

from model import GPT, GPTConfig
from tokenizer import CharTokenizer

OUT_DIR = os.environ.get("JO_OUT_DIR", "out")
BLIP_NAME = "Salesforce/blip-image-captioning-base"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}
MAX_FILE_CHARS = 2000  # Jo only sees its last 256 characters anyway

device = "cuda" if torch.cuda.is_available() else "cpu"
app = Flask(__name__, static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024  # 25 MB per request
gpu_lock = threading.Lock()  # one generation at a time on the GPU

# ---------------- Jo ----------------
jo = {"model": None, "tok": None, "mtime": None, "info": {}}


def load_jo():
    """(Re)load Jo's checkpoint. Reloads automatically when train.py saves a newer one."""
    path = os.path.join(OUT_DIR, "ckpt.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No checkpoint at {path}. Run `python train.py` first.")
    mtime = os.path.getmtime(path)
    if jo["mtime"] != mtime:
        ckpt = torch.load(path, map_location=device)
        model = GPT(GPTConfig(**ckpt["config"])).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        jo.update(model=model, tok=CharTokenizer.load(os.path.join(OUT_DIR, "tokenizer.json")), mtime=mtime,
                  info={"iter": ckpt.get("iter"), "val_loss": ckpt.get("val_loss"),
                        "params_m": round(model.num_params() / 1e6, 2)})
        print(f"Loaded Jo from step {ckpt.get('iter')} (val loss {ckpt.get('val_loss'):.4f})")
    return jo["model"], jo["tok"]


# ---------------- BLIP (image → caption) ----------------
blip = {"model": None, "proc": None}


def caption_image(img):
    if blip["model"] is None:  # loaded on first use; the first time downloads ~1 GB from Hugging Face
        from transformers import BlipForConditionalGeneration, BlipProcessor
        print("Loading BLIP captioning model...")
        blip["proc"] = BlipProcessor.from_pretrained(BLIP_NAME)
        blip["model"] = BlipForConditionalGeneration.from_pretrained(
            BLIP_NAME, torch_dtype=torch.float16 if device == "cuda" else torch.float32).to(device).eval()
    inputs = blip["proc"](images=img.convert("RGB"), return_tensors="pt").to(device, blip["model"].dtype)
    with torch.no_grad():
        out = blip["model"].generate(**inputs, max_new_tokens=40)
    return blip["proc"].decode(out[0], skip_special_tokens=True).strip()


# ---------------- routes ----------------
@app.get("/")
def index():
    return send_from_directory("static", "index.html")


@app.get("/api/status")
def status():
    try:
        load_jo()
        return jsonify(ok=True, device=device, blip_loaded=blip["model"] is not None, **jo["info"])
    except FileNotFoundError as e:
        return jsonify(ok=False, error=str(e)), 503


@app.post("/api/generate")
def generate():
    text = request.form.get("prompt", "")
    tokens = max(1, min(int(request.form.get("tokens", 400)), 3000))
    temperature = max(0.05, min(float(request.form.get("temperature", 0.8)), 2.0))
    top_k = max(1, min(int(request.form.get("top_k", 40)), 65))
    files = request.files.getlist("files")
    # read uploads now: the request body is gone once the streaming response starts
    uploads = [(f.filename or "file", f.read()) for f in files]

    def events():
        def ev(**kw):
            return json.dumps(kw) + "\n"

        try:
            model, tok = load_jo()
        except FileNotFoundError as e:
            yield ev(type="error", message=str(e))
            return

        parts = [text] if text.strip() else []
        for name, data in uploads:
            ext = os.path.splitext(name)[1].lower()
            try:
                if ext in IMAGE_EXTS:
                    yield ev(type="status", message=f"Looking at {name}…")
                    with gpu_lock:
                        cap = caption_image(Image.open(io.BytesIO(data)))
                    yield ev(type="caption", file=name, caption=cap)
                    parts.append(cap[:1].upper() + cap[1:] + ".")
                else:
                    content = data.decode("utf-8", errors="replace")
                    if content.count("�") > len(content) * 0.05:
                        yield ev(type="warning", message=f"{name} doesn't look like a text file — skipped.")
                        continue
                    parts.append(content[-MAX_FILE_CHARS:])
                    yield ev(type="file", file=name, chars=len(content))
            except Exception as e:  # a bad file shouldn't kill the whole request
                yield ev(type="warning", message=f"Couldn't read {name}: {e}")

        prompt = "\n".join(parts)
        if prompt and not prompt.endswith("\n"):
            prompt += "\n"
        ids = tok.encode(prompt) or tok.encode("\n")
        # show exactly what Jo sees: unknown characters (emoji, accents, ...) are dropped by the tokenizer
        seen = tok.decode(ids[-model.cfg.block_size:])
        yield ev(type="prompt", text=seen, dropped=len(prompt) - len(ids), truncated=len(ids) > model.cfg.block_size)

        with gpu_lock:
            idx = torch.tensor([ids], dtype=torch.long, device=device)
            buf = []
            for next_id in model.generate_stream(idx, tokens, temperature=temperature, top_k=top_k):
                buf.append(next_id.item())
                if len(buf) >= 4:  # send a few characters per chunk
                    yield ev(type="token", text=tok.decode(buf))
                    buf = []
            if buf:
                yield ev(type="token", text=tok.decode(buf))
        yield ev(type="done")

    return Response(stream_with_context(events()), mimetype="application/x-ndjson")


# ---------------- "My code": style-tuned code model (code_style.py) ----------------
NOTES_DIR = os.environ.get("JO_NOTES_DIR", os.path.join("private", "notes"))
coder = {"model": None, "tok": None}


def free_gpu(keep):
    """The 6 GB laptop GPU can't comfortably hold every model, so unload the big ones we're not using."""
    import notes_rag
    if keep != "coder" and coder["model"] is not None:
        coder.update(model=None, tok=None)
    if keep != "notes":
        notes_rag._models.pop("chat", None)
    if device == "cuda":
        torch.cuda.empty_cache()


def ndjson(gen):
    return Response(stream_with_context((json.dumps(e) + "\n" for e in gen)), mimetype="application/x-ndjson")


@app.post("/api/code")
def code():
    import code_style
    prompt = request.form.get("prompt", "")
    tokens = max(16, min(int(request.form.get("tokens", 200)), 1024))
    temperature = max(0.0, min(float(request.form.get("temperature", 0.4)), 1.5))
    use_adapter = request.form.get("base") != "1"

    def events():
        if not os.path.isdir(code_style.ADAPTER_DIR):
            yield dict(type="error", message="No style adapter yet. Train one: "
                                             "python code_style.py train --code_dir \"path\\to\\your\\code\"")
            return
        with gpu_lock:
            if coder["model"] is None:
                yield dict(type="status", message="Loading the code model…")
                free_gpu("coder")
                coder["model"], coder["tok"], _ = code_style.load_model()
            yield dict(type="status", message="")
            for piece in code_style.complete_stream(coder["model"], coder["tok"], device, prompt,
                                                    tokens, temperature, use_adapter):
                yield dict(type="token", text=piece)
        yield dict(type="done")

    return ndjson(events())


# ---------------- "My notes": private RAG over private/notes (notes_rag.py) ----------------
@app.post("/api/ask")
def ask():
    import notes_rag
    question = request.form.get("prompt", "").strip()
    k = max(1, min(int(request.form.get("k", 4)), 10))

    def events():
        if not os.path.isdir(NOTES_DIR) or not notes_rag.note_files(NOTES_DIR):
            yield dict(type="error", message=f"No notes found. Put .txt or .md files in {NOTES_DIR}.")
            return
        with gpu_lock:
            yield dict(type="status", message="Searching your notes…")
            hits, rebuilt, index = notes_rag.ask(question, NOTES_DIR, k)
            if rebuilt:
                yield dict(type="info", message=f"Re-indexed {len(index['files'])} notes ({len(index['chunks'])} chunks).")
            yield dict(type="sources", hits=hits)
            if notes_rag._models.get("chat") is None:
                yield dict(type="status", message="Loading the answer model…")
                free_gpu("notes")
            yield dict(type="status", message="")
            for piece in notes_rag.answer_stream(question, hits):
                yield dict(type="token", text=piece)
        yield dict(type="done")

    return ndjson(events())


@app.get("/api/modes")
def modes():
    import code_style
    import notes_rag
    meta_path = os.path.join(code_style.ADAPTER_DIR, "jo_meta.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else None
    n_notes = len(notes_rag.note_files(NOTES_DIR)) if os.path.isdir(NOTES_DIR) else 0
    return jsonify(code=meta, notes={"dir": NOTES_DIR, "files": n_notes})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("JO_PORT", 5000)), threaded=True)
