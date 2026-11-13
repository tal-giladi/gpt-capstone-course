"""Gradio chat demo for your own GPT (lesson 11.2).

Runs unchanged on a ZeroGPU Space, a CPU Space, any VPS, or your laptop:

    MODEL_ID=you/my-gpt-350m python app.py              # a Hub repo (public, or private with HF_TOKEN)
    MODEL_ID=../../hf/my-gpt-350m-dpo python app.py     # a local folder from export_hf / dpo.py

Settings (environment variables, all optional except MODEL_ID):
    MODEL_ID          Hub repo id or local folder (required)
    MODEL_REVISION    commit sha, tag or branch to pin (default: main)
    MAX_NEW_TOKENS    hard cap on answer length (default 256)
    RATE_LIMIT        requests per client per RATE_WINDOW seconds (default 20 per 600)
    MAX_INPUT_CHARS   longest accepted message (default 2000)

Nothing the visitor types is written to disk or logged by this app.
"""
from __future__ import annotations

import os
import threading
import time
from collections import defaultdict, deque

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

try:  # present on Hugging Face Spaces; @spaces.GPU is a no-op outside ZeroGPU
    import spaces
    gpu = spaces.GPU(duration=60)
except ImportError:  # laptop, VPS
    def gpu(fn):
        return fn

MODEL_ID = os.environ.get("MODEL_ID", "").strip()
if not MODEL_ID:
    raise SystemExit("Set MODEL_ID to your Hub repo (you/my-gpt-350m) or a local model folder.")
REVISION = os.environ.get("MODEL_REVISION") or None
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "256"))
RATE_LIMIT = int(os.environ.get("RATE_LIMIT", "20"))
RATE_WINDOW = int(os.environ.get("RATE_WINDOW", "600"))
MAX_INPUT_CHARS = int(os.environ.get("MAX_INPUT_CHARS", "2000"))
MAX_TURNS = 6  # keep only the last few exchanges; a 1,024-token context fills up fast

DISCLAIMER = (
    "**Research demo.** A ~350M-parameter model trained from scratch as a learning project. "
    "It makes up facts, gets maths wrong and can produce biased or offensive text. "
    "Do not rely on its answers and do not enter personal or confidential information. "
    "Messages are not stored by this app."
)

# ---- model (loaded once, at import time) ---------------------------------------------------
device = "cuda" if torch.cuda.is_available() or os.environ.get("SPACES_ZERO_GPU") else "cpu"
if device == "cpu":
    torch.set_num_threads(max(1, os.cpu_count() or 1))
dtype = torch.bfloat16 if device == "cuda" else torch.float32  # fp32 is faster than bf16 on most CPUs
tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=REVISION, dtype=dtype).to(device).eval()
CONTEXT = getattr(model.config, "max_position_embeddings", 1024)
CHAT = bool(getattr(tok, "chat_template", None))
print(f"[app] {MODEL_ID}@{REVISION or 'main'} on {device} ({dtype}), context {CONTEXT}, chat={CHAT}")

# ---- per-client rate limit (in memory; resets when the Space restarts) ----------------------
_hits: dict[str, deque] = defaultdict(deque)
_lock = threading.Lock()


def client_id(request: gr.Request | None) -> str:
    if request is None:
        return "local"
    fwd = (request.headers or {}).get("x-forwarded-for", "")
    return fwd.split(",")[0].strip() or (request.client.host if request.client else "unknown")


def allow(cid: str) -> bool:
    now = time.time()
    with _lock:
        q = _hits[cid]
        while q and now - q[0] > RATE_WINDOW:
            q.popleft()
        if len(q) >= RATE_LIMIT:
            return False
        q.append(now)
        return True


# ---- prompt building -----------------------------------------------------------------------
def text_of(content) -> str:
    """Gradio message content can be a string or a list of parts; keep the text."""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return str(content.get("text") or content.get("content") or "")
    if isinstance(content, (list, tuple)):
        return " ".join(text_of(c) for c in content)
    return str(content or "")


def build_ids(message: str, history: list, max_new: int) -> torch.Tensor:
    budget = CONTEXT - max_new  # prompt + answer must fit in the context window
    turns = [{"role": h["role"], "content": text_of(h["content"])} for h in history
             if isinstance(h, dict) and h.get("role") in ("user", "assistant")][-2 * MAX_TURNS:]
    while True:
        if CHAT:
            msgs = turns + [{"role": "user", "content": message}]
            ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt",
                                          return_dict=True)["input_ids"]
        else:  # base model: plain continuation of the message
            ids = tok(message, return_tensors="pt")["input_ids"]
        if ids.shape[1] <= budget or not turns:
            return ids[:, -budget:]  # last resort: keep the end of an over-long single message
        turns = turns[2:]  # drop the oldest exchange and try again


# ---- generation ----------------------------------------------------------------------------
@gpu  # on ZeroGPU this generator runs with a GPU attached, in a separate worker process
def stream_answer(ids: torch.Tensor, max_new: int, temperature: float):
    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True, timeout=120)
    kwargs = dict(input_ids=ids.to(device), streamer=streamer, max_new_tokens=max_new,
                  do_sample=temperature > 0, temperature=temperature if temperature > 0 else None,
                  top_p=0.9 if temperature > 0 else None, repetition_penalty=1.1,
                  eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id or tok.eos_token_id)

    def run():
        with torch.no_grad():
            model.generate(**kwargs)

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    text = ""
    for piece in streamer:
        text += piece
        yield text
    worker.join()


def respond(message, history, max_new_tokens, temperature, request: gr.Request = None):
    # validation and rate limiting stay in the main process (ZeroGPU forks for stream_answer)
    message = text_of(message).strip()
    if not message:
        yield "Type a message first."
        return
    if len(message) > MAX_INPUT_CHARS:
        yield f"Please keep messages under {MAX_INPUT_CHARS} characters."
        return
    if not allow(client_id(request)):
        yield f"Rate limit: at most {RATE_LIMIT} messages per {RATE_WINDOW // 60} minutes. Please wait a bit."
        return
    max_new = max(1, min(int(max_new_tokens), MAX_NEW_TOKENS))
    ids = build_ids(message, history or [], max_new)
    text = ""
    for text in stream_answer(ids, max_new, float(temperature)):
        yield text
    if not text.strip():
        yield "(the model returned an empty answer; try rephrasing)"


demo = gr.ChatInterface(
    respond,
    title=f"Chat with {MODEL_ID.rstrip('/').split('/')[-1]}",
    description=DISCLAIMER,
    additional_inputs=[
        gr.Slider(16, MAX_NEW_TOKENS, value=min(160, MAX_NEW_TOKENS), step=16, label="Max new tokens"),
        gr.Slider(0.0, 1.2, value=0.7, step=0.1, label="Temperature (0 = greedy)"),
    ],
    examples=[["Explain photosynthesis in two sentences.", 160, 0.7],
              ["Give me three tips for learning a new language.", 160, 0.7],
              ["What is the difference between weather and climate?", 160, 0.7]],
    cache_examples=False,
    flagging_mode="never",
    concurrency_limit=1,  # one generation at a time; others wait in the queue
)
demo.queue(max_size=16)

if __name__ == "__main__":
    demo.launch()
