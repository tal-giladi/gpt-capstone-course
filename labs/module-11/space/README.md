---
title: My GPT 350M chat
emoji: 💬
colorFrom: indigo
colorTo: green
sdk: gradio
sdk_version: 6.32.0
python_version: "3.12"
app_file: app.py
pinned: true
license: apache-2.0
short_description: A 350M GPT trained from scratch, SFT + DPO. Research demo.
models:
  - your-username/my-gpt-350m
---

# My GPT 350M — chat demo

A research demo of a ~350M-parameter GPT trained from scratch on FineWeb-Edu as a learning
project, then instruction-tuned (SFT) and preference-tuned (DPO). Model card, evaluation and
limitations: https://huggingface.co/your-username/my-gpt-350m

The model makes up facts and can produce biased or offensive text. Do not rely on its answers and
do not enter personal information. The app does not store messages.

## Settings (Space → Settings → Variables)

| Variable | Example | Meaning |
|---|---|---|
| `MODEL_ID` | `your-username/my-gpt-350m` | Model repo to load (required) |
| `MODEL_REVISION` | `v1.0` | Tag, branch or commit to pin |
| `MAX_NEW_TOKENS` | `256` | Hard cap on answer length |
| `RATE_LIMIT` / `RATE_WINDOW` | `20` / `600` | Messages per visitor per window (seconds) |

If the model repo is private or gated, add a **secret** `HF_TOKEN` with a fine-grained read token
scoped to that one repo.

## Run locally

```bash
pip install gradio==6.32.0 -r requirements.txt
MODEL_ID=your-username/my-gpt-350m python app.py     # then open http://127.0.0.1:7860
```

Replace `your-username` in the YAML header and above before pushing.
