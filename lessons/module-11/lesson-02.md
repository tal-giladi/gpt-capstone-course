---
id: "11.2"
module: 11
minutes: 25
practice_minutes: 90
prerequisites: ["11.1"]
objectives:
  - Compare demo hosts (Hugging Face Spaces on ZeroGPU, CPU and paid GPUs, Modal, Replicate, RunPod Serverless, a Hetzner VPS) on current price, limits and cold-start behaviour, and pick one that keeps the demo at $0.
  - Estimate CPU generation speed for your model from a local measurement and decide whether CPU hosting is usable.
  - Run the Gradio chat app locally against your release, with streaming, a rate limit, input and length caps and a disclaimer.
  - Deploy it as a public Space pinned to your v1.0 model and check it from a logged-out browser.
volatility: implementation
sources:
  - title: "Spaces overview (Gradio/Docker Spaces need a paid plan to create; hardware; secrets)"
    url: https://huggingface.co/docs/hub/spaces-overview
  - title: "Spaces ZeroGPU (free hosting for eligible accounts, quotas, supported versions)"
    url: https://huggingface.co/docs/hub/spaces-zerogpu
  - title: "Using GPU Spaces (hardware prices, billing, sleep time, pausing)"
    url: https://huggingface.co/docs/hub/spaces-gpus
  - title: "Spaces configuration reference (README YAML)"
    url: https://huggingface.co/docs/hub/spaces-config-reference
  - title: "Hugging Face pricing (PRO $9/month, Spaces hardware)"
    url: https://huggingface.co/pricing
  - title: "Modal pricing"
    url: https://modal.com/pricing
  - title: "Replicate pricing"
    url: https://replicate.com/pricing
  - title: "RunPod pricing (Serverless)"
    url: https://www.runpod.io/pricing
  - title: "Hetzner Cloud regular-performance (CPX) prices"
    url: https://www.hetzner.com/cloud/regular-performance/
last_verified: "2026-11-13"
---

# 11.2 · A live demo Space

A model page tells people your model exists; a demo lets them find out in ten seconds what it can and cannot do. For a 350M model the demo has to be honest (it will say silly things, and the page should say so before it does), cheap (nobody should pay a monthly bill for a portfolio piece) and robust against the one visitor who pastes a novel or scripts a thousand requests. You compare the hosting options at today's prices, measure how fast your model is on a CPU, run the provided Gradio app locally against your release, and deploy it as a public Space that loads `you/my-gpt-350m@v1.0`.

## Goal

A public Hugging Face Space `<you>/my-gpt-350m-chat` running `labs/module-11/space/app.py`: a chat page that streams answers from your DPO model at tag `v1.0`, shows a disclaimer above the chat, caps message length and answer length, limits each visitor to 20 messages per 10 minutes, and costs $0 per month. Your notes record the measured CPU speed and the host you chose with the reason.

## What you need

- The public repo `<you>/my-gpt-350m` with tag `v1.0` and the local `release/my-gpt-350m/` folder from [11.1](lesson-01.md).
- Your Hugging Face account with a **verified email, older than 30 days** (you created it in 01.2; check the date on your profile settings). This decides which free option you can use.
- The [Module 11 lab files](../../labs/module-11/): `space/` and `cpu_speed.py` copied into your repo, and the project environment with `pip install -e ".[demo]"` (gradio 6.32).
- About 2 hours. **Expected cost: $0** (or $9 if you choose the PRO fallback below).

## Choices and costs

### How fast is your model on a CPU?

Measured on 2026-11-13 with `cpu_speed.py`'s method on a randomly initialised model with your exact shape (337M parameters, Qwen3 layout, 200-token prompt, 128 new tokens, greedy), on a laptop CPU (Intel Core Ultra 7 255H):

| Threads | dtype | Tokens/s | 160-token answer |
|---|---|---|---|
| 2 | fp32 | 6.3 | ~25 s |
| 2 | bf16 | 4.4 | ~36 s |
| 8 | fp32 | 4.1 (noisy: hybrid laptop cores) | ~39 s |

Two shared cloud vCPUs are usually slower than two laptop performance cores, so plan for **~3–6 tokens/s** on a 2-vCPU host (ESTIMATE): an answer streams for 30–50 seconds and a second visitor waits for the first. Usable for a portfolio demo with streaming, not pleasant. fp32 beats bf16 on CPUs without native bf16 matrix units, which is why `app.py` uses fp32 on CPU. On a GPU the same model generates well over 50 tokens/s (ESTIMATE; `generate` is limited by Python overhead at this size), so an answer takes a few seconds.

### Where the demo runs (prices checked 2026-11-13)

| Option | Price | Limits and behaviour | Fit |
|---|---|---|---|
| **HF Space on ZeroGPU** | **$0** for free accounts in good standing (verified email, account > 30 days), up to 2 such Spaces; PRO can host 10 | Gradio SDK only; GPU (half an NVIDIA RTX Pro 6000, 48 GB) attached per call; visitors spend **their own** daily quota: 2 min logged out, 5 min free account, 40 min PRO; supported torch 2.8–2.13, Python 3.10/3.12 | **Recommended.** Free, fast answers; a 350M answer costs a few seconds of a visitor's quota |
| HF Space on CPU Basic | Hardware free (2 vCPU, 16 GB RAM, 50 GB ephemeral disk), but **creating a Gradio or Docker Space now requires a paid plan**: PRO $9/month | Sleeps after 48 h without visitors and wakes on the next visit (first load waits for the start); ~3–6 tokens/s | Fallback if your account is too young for ZeroGPU: $9 for one month, then cancel PRO (check that the Space keeps running after cancelling before you rely on it) |
| HF Space CPU Upgrade | $0.03/h (~$22/month), 8 vCPU, 32 GB | Never sleeps by default; set a sleep time | Faster CPU, but a monthly bill for a demo |
| HF Space T4 small | $0.40/h (~$288/month if always on) | Custom sleep time stops billing while idle | Not needed for 350M |
| Modal | Starter plan includes $30/month of compute; CPU ~$0.047/core-h, memory ~$0.008/GiB-h, T4 $0.000164/s (~$0.59/h); scales to zero | Your own web endpoint; cold start on the first request after idle | Good if you want an API, not just a page; stays at $0 inside the credit |
| Replicate | CPU small $0.09/h, T4 $0.81/h; private models are billed for setup and idle time while online | Needs a Cog container | More work, and idle billing for a private model |
| RunPod Serverless | 16 GB GPU $0.58/h, 24 GB (L4/A5000) $0.75/h; no free tier | Scale-to-zero workers, cold starts; you build the container and the front end | Overkill for a chat page |
| Hetzner Cloud VPS | CPX22 (2 vCPU, 4 GB, Helsinki) €16.99/month; CPX12 (1 vCPU, 2 GB) €9.99 is too small for fp32 weights + torch | Always on, you run updates, TLS and the firewall yourself | Only if you want full control; not $0 |
| Local only + a screen recording | $0 | Nobody can try it | Last resort; link the video from the card |

> [!NOTE]
> Hugging Face changed the Spaces rules in 2026 (free accounts can no longer create CPU Gradio Spaces; ZeroGPU moved to RTX Pro 6000). Re-check the Spaces overview and ZeroGPU pages before you create the Space, and record the option and price you actually used in `budget.md`.

### Gradio or Streamlit

| | Gradio 6.32 | Streamlit |
|---|---|---|
| On Spaces | Native SDK (`sdk: gradio`) | Not a native SDK any more: only through a Docker Space |
| ZeroGPU | Yes (the only supported SDK) | No |
| Chat UI with streaming | `gr.ChatInterface` with a generator function | `st.chat_message` + `st.write_stream`, more code |
| Queue, concurrency limit, API | Built in (`queue(max_size)`, `concurrency_limit`, auto API) | You build it |

Use Gradio. `app.py` is about 170 lines and works unchanged on ZeroGPU, CPU, a VPS and your laptop.

### What the app does to stay safe and cheap

| Risk | Control in `app.py` |
|---|---|
| Visitors trust wrong answers | Disclaimer above the chat: research model, invents facts, may be offensive, do not enter personal data |
| One visitor monopolises the demo | `RATE_LIMIT` messages per `RATE_WINDOW` seconds per client IP (default 20 per 10 min), in memory |
| Huge inputs or answers | Messages over 2,000 characters refused; answer capped by `MAX_NEW_TOKENS` (256); old turns dropped to fit the 1,024-token context |
| Queue pile-up | `concurrency_limit=1`, queue of 16 |
| Privacy | Nothing is logged or stored; flagging is off |
| Secrets | No token needed for a public model; if you ever load a private one, use a fine-grained **read** token scoped to that repo as a Space **secret**, never a variable or a file |

Gradio also exposes the chat as an API endpoint (`/respond`); the same rate limit and caps apply to it, because they run inside the function.

## Do this

### 1. Measure your real model on a CPU (10 min)

```bash
python tools/cpu_speed.py --model release/my-gpt-350m --threads 2
python tools/cpu_speed.py --model release/my-gpt-350m --threads 2 --dtype bf16
```

Expect numbers close to the table above (your model has the same shape). Write both into `reports/demo.md`.

### 2. Run the app locally against your release (15 min)

```bash
# copy the lab's space/ folder to the root of your project repo first: <lab download>/space -> ./space
MODEL_ID=release/my-gpt-350m python space/app.py
```

Open http://127.0.0.1:7860. The console prints `[app] release/my-gpt-350m@main on cpu (torch.float32), context 1024, chat=True`. Try the three examples, then the controls:

- Send a 3,000-character message: refused with "Please keep messages under 2000 characters."
- Restart with `RATE_LIMIT=3` and send four messages: the fourth is refused with the rate-limit text.
- Hold a long conversation (8+ turns): it keeps answering, because the oldest turns are dropped to fit the context.

Stop it with Ctrl-C.

### 3. Fill in the Space README (5 min)

In `space/README.md` replace `your-username` (YAML `models:` entry and the text). Keep `sdk: gradio`, `sdk_version: 6.32.0` and `python_version: "3.12"`: Spaces installs Gradio from `sdk_version`, and ZeroGPU supports Python 3.12 and torch up to 2.13.0, which is why `requirements.txt` pins `torch==2.13.0` rather than the 2.15 you train with.

### 4. Create the Space and push (15 min)

Check your eligibility first: Settings → Account shows the creation date; the email must be verified.

```bash
U=<your username>
hf repos create $U/my-gpt-350m-chat --repo-type space --sdk gradio
hf upload $U/my-gpt-350m-chat space . --repo-type space --commit-message "Chat demo for my-gpt-350m v1.0"
```

Then in the browser, on the Space's **Settings** page:

1. **Hardware:** choose **ZeroGPU**. (With PRO instead: leave CPU basic.)
2. **Variables and secrets → New variable:** `MODEL_ID` = `<you>/my-gpt-350m`, `MODEL_REVISION` = `v1.0`. Variables are public, which is fine for these values.
3. Leave visibility **public**.

The Space builds (installing torch takes a few minutes) and starts. The **Logs** tab must show the `[app]` line with `on cuda` for ZeroGPU (or `on cpu` on CPU basic).

> [!WARNING]
> Never choose paid Space hardware "just to try it": a T4 small runs at $0.40/h and an upgraded Space never sleeps by default, so it bills every hour until you downgrade or pause it. If you do upgrade, set a sleep time in the same session and add a calendar reminder to downgrade.

If you host somewhere else, the same `app.py` runs unchanged; only the setup differs:

| Host | What to do instead of step 4 |
|---|---|
| HF Space on CPU basic (PRO) | Same commands; leave hardware on CPU basic |
| Hetzner (or any) VPS | `pip install gradio==6.32.0 -r space/requirements.txt` (torch 2.15.0 is fine here), then `MODEL_ID=<you>/my-gpt-350m MODEL_REVISION=v1.0 GRADIO_SERVER_NAME=0.0.0.0 python space/app.py` inside tmux; open port 7860 in the firewall, or better put a reverse proxy with HTTPS in front |
| Modal, Replicate, RunPod Serverless | Not provided by the lab: each needs its own wrapper (a Modal web endpoint, a Cog container, a serverless handler) around `app.py`'s `respond` function. Budget an extra 1–2 hours and use their scale-to-zero setting |
| Local only | `python space/app.py` and record the screen; link the video from the card |

### 5. Test like a visitor (10 min)

Open `https://huggingface.co/spaces/<you>/my-gpt-350m-chat` in a private window (logged out). Ask the three examples and one question from your probe set where you know the model fails. Note the time to first token and to the full answer. Logged out you have 2 minutes of ZeroGPU per day, enough for many short answers; if the page says the quota is exhausted, log in or wait.

### 6. Link everything (10 min)

Add the Space URL to both model cards (the "Demo" row), re-upload the README with `hf upload $U/my-gpt-350m release/my-gpt-350m/README.md README.md`, and add to `reports/demo.md`: host, hardware, measured speeds, why you chose it, the Space URL and the `v1.0` it loads. Commit `space/` and `reports/demo.md`.

## How you know it worked

- A logged-out browser shows the disclaimer, streams an answer token by token, and the Space page lists your model under "Models used".
- The Space's logs show your repo id with `@v1.0`; changing `main` later does not change the demo.
- Locally, the 3,000-character message and the fourth message with `RATE_LIMIT=3` were refused with the app's messages.
- The Space's Settings page shows **ZeroGPU** (or CPU basic) and no paid hardware; your HF billing page shows $0 for Spaces (or the $9 PRO line if you chose the fallback).

## When it goes wrong

| Symptom | Cause | Fix |
|---|---|---|
| "Creating a Gradio Space requires a paid plan" | Account younger than 30 days or email unverified, so ZeroGPU hosting is not open to you yet | Wait until the account is 30 days old, or use the PRO fallback ($9) on CPU basic, or run Modal within its free credit |
| Build fails: no matching distribution for `torch==2.13.0` | `python_version` not 3.10/3.12, or a typo in `requirements.txt` | Keep `python_version: "3.12"`; on CPU basic you may use `torch==2.15.0` instead |
| Runtime error `Set MODEL_ID ...` | Variable missing or named differently | Add the `MODEL_ID` variable and restart the Space |
| `401`/`Repository Not Found` in the logs | The model repo is still private | Make it public (11.1), or add a fine-grained read token as secret `HF_TOKEN` |
| `ZeroGPU quota exceeded` for every visitor | `MAX_NEW_TOKENS` high and long answers, or a scripted client | Lower `MAX_NEW_TOKENS` to 160 and `RATE_LIMIT`; quotas reset 24 h after first use |
| Answers never start on CPU basic | The Space was asleep and is loading the 1.35 GB model, or a long answer from another visitor is in the queue | Wait for the start (the page shows "Building"/"Starting"); lower `MAX_NEW_TOKENS` |
| Gibberish, or the chat tags appear in the answer | `MODEL_ID` points at the base model, or the chat template was not uploaded | Point it at `<you>/my-gpt-350m` and check `chat_template.jinja` in the model repo |
| The demo got worse after you pushed a new model | `MODEL_REVISION` empty, so it follows `main` | Set `MODEL_REVISION=v1.0` and restart |

## Artifact

- Public Space `<you>/my-gpt-350m-chat` serving `<you>/my-gpt-350m@v1.0`, linked from both model cards.
- In git: `space/` (app, pinned requirements, README with YAML header) and `reports/demo.md` (measured CPU tokens/s, host decision with today's prices, URL).
- `budget.md` line:

  ```text
  | 2026-11-13 | 11.2 | HF Space on ZeroGPU (free tier), local CPU tests | $0.00 | $0.00 |
  ```
