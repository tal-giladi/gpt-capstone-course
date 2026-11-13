---
id: "09.1"
module: 9
minutes: 20
practice_minutes: 50
prerequisites: ["03.1", "04.2", "08.1"]
objectives:
  - Choose an open SFT dataset for a ~350M model with a 1,024-token context and justify it by license, size, conversation length and what it was built for.
  - Render your exported model's chat template and show which tokens the SFT loss trains on.
  - Measure the dataset through your own tokenizer (tokens per conversation, share truncated, trained-token share) and turn it into GPU-hours and dollars.
  - Pick the box for SFT and DPO and write the budget line before renting it.
volatility: implementation
sources:
  - title: "HuggingFaceTB/smol-smoltalk dataset card"
    url: https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk
  - title: "HuggingFaceTB/smoltalk dataset card"
    url: https://huggingface.co/datasets/HuggingFaceTB/smoltalk
  - title: "HuggingFaceTB/smoltalk2 dataset card"
    url: https://huggingface.co/datasets/HuggingFaceTB/smoltalk2
  - title: "allenai/tulu-3-sft-mixture dataset card"
    url: https://huggingface.co/datasets/allenai/tulu-3-sft-mixture
  - title: "teknium/OpenHermes-2.5 dataset card"
    url: https://huggingface.co/datasets/teknium/OpenHermes-2.5
  - title: "SmolLM2: When Smol Goes Big (paper)"
    url: https://arxiv.org/pdf/2502.02737
  - title: "alignment-handbook SmolLM2 recipes (SFT/DPO configs for 135M/360M)"
    url: https://github.com/huggingface/alignment-handbook/tree/main/recipes/smollm2
  - title: "TRL 1.17 SFT Trainer docs (assistant_only_loss, packing, max_length)"
    url: https://huggingface.co/docs/trl/sft_trainer
last_verified: "2026-11-13"
---

# 09.1 · Choosing SFT data and a chat template

Your base model continues text; it does not answer questions. Supervised fine-tuning (SFT) teaches it one new habit: when it sees a user turn in a fixed format, write a helpful answer and then stop. Which conversations you show it, how they are formatted and which tokens the loss looks at decide most of what the chat model will be like, and none of that costs money to get right. This lesson is laptop work: you choose the dataset, look at your chat template token by token, measure the data through your own tokenizer, and only then price the GPU time for 09.2.

## Goal

You have a written data decision in `reports/sft_data.md`: the SFT dataset and its license, the context length and number of epochs, measured statistics of the data through your 32k tokenizer and chat template (tokens per conversation, share truncated, share of tokens trained on), the resulting estimate of GPU-hours and dollars, and the box you will rent in 09.2. You have also seen, token by token, that the loss trains only on assistant answers and on the `<|endoftext|>` that ends each one.

## What you need

- Your exported base model in the bucket under `hf/my-gpt-350m-base/` (from Module 8), and `r2.env` from [02.2](../module-02/lesson-02.md) on your laptop.
- Your project repo with the extras installed on your laptop: `pip install -e ".[post]"` (TRL, PEFT, accelerate, judge SDKs; CPU is enough for this lesson).
- The [Module 9 lab files](../../labs/module-09/) copied into `tools/` and committed.
- About 70 minutes, ~1.4 GB of disk for the model and a few hundred MB of dataset cache.
- **Expected cost: $0.** Everything runs on your laptop; the datasets are free downloads from the Hugging Face Hub.

## Choices and costs

### The dataset

All of these are free to download. What differs is license, size, conversation length and how hard the content is for a 350M model trained on 20B tokens.

| Dataset (checked 2026-11-13) | Size | License | Built for | Fit for your model |
|---|---|---|---|---|
| **`HuggingFaceTB/smol-smoltalk`** | 460k train / 24k test conversations, ~971 MB | Apache-2.0 | Models under 1B; the SFT data of SmolLM2-360M-Instruct and -135M-Instruct | **Recommended.** Shorter Magpie-Ultra conversations, no function calling, no advanced math. Already in the `messages` format `sft.py` expects, with a `test` split for eval loss |
| `HuggingFaceTB/smoltalk` (config `all`) | 1.04M train / 55k test | Apache-2.0 for the new subsets; included public datasets keep their own licenses | SmolLM2-1.7B-Instruct | Twice the data, but longer conversations, advanced math, rewriting and function calling that a 350M model mostly cannot learn |
| `HuggingFaceTB/smoltalk2` (SFT subset) | 3.38M rows, ~1.5M with reasoning traces (`think`) | Apache-2.0 for new data; sources keep theirs | SmolLM3-3B, long context (subsets up to 64k tokens) | Too long and too hard for a 1,024-token, 350M model; reasoning traces would be truncated |
| `allenai/tulu-3-sft-mixture` | 939k rows, 1.41 GB | ODC-BY-1.0, research/education; some subsets differ (No Robots is CC-BY-NC-4.0) | Llama-3.1-Tulu-3-8B | Heavy on math, code and safety data for an 8B model; the non-commercial subset matters if you publish |
| `teknium/OpenHermes-2.5` | ~1M rows | No license stated on the card; GPT-4-generated | 7B community models | Different format (`conversations` with `from`/`value`, needs conversion) and unclear terms for a public model; avoid |

Why the recommendation is not a close call: smol-smoltalk was assembled for exactly your size class, it is the data behind a published 360M chat model, its license lets you publish the result in Module 11, and `scripts/sft.py` uses it by default. The published recipe for SmolLM2-360M-Instruct (alignment-handbook `config_smol.yaml`) trains it for **2 epochs** with a cosine schedule and 10% warmup; `sft.py` defaults to the same 2 epochs.

> [!IMPORTANT]
> Set your expectations from that reference point. SmolLM2-360M-Instruct was pretrained on **4 trillion** tokens; your model saw about **20 billion**, 200× fewer. With the same SFT data, expect a model that writes fluent, well-formatted short answers and stops on time, but gets facts and arithmetic wrong far more often than SmolLM2 does. 09.2 shows what that looks like.

### Context length and truncation

Your model's context is `block_size` from `configs/main_350m.yaml`: 1,024 unless your context ablation in Module 6 paid off and you trained at 2,048. SFT cannot go beyond it. Measured on 3,000 smol-smoltalk conversations through the chat template (GPT-2 tokenizer, 2026-11-13; your 32k tokenizer gives similar counts, and step 4 measures them exactly):

| Statistic | `--max-length 1024` | `--max-length 2048` |
|---|---|---|
| Tokens per conversation (mean / median / p90) | 955 / 927 / 1,395 | same |
| Conversations longer than the limit (truncated) | **43%** | 1.6% |
| Conversations dropped because truncation removed every assistant token | 1.7% | 1.1% |
| Share of kept tokens that are trained (assistant) tokens | 75% | 77% |
| Tokens per epoch after truncation | ~334M | ~390M |

TRL truncates with `keep_start`: a long conversation keeps its first turns and loses the end. That is acceptable. The model still sees complete first exchanges, which is what most chat use looks like; only the late turns of long conversations are lost. Do not filter out long conversations to "fix" it: you would throw away 43% of the data and bias the model towards short answers.

### Packing, epochs, tokens

| Decision | Choice | Why |
|---|---|---|
| Epochs | **2** (`sft.py` default, as in the SmolLM2 recipe) | ~670M trained-on tokens, about 3% of pretraining. One epoch leaves the format less stable; three starts to memorize answers |
| Packing | **Off** (default) | `--packing` in TRL 1.17 uses padding-free batches, which the TRL docs say need FlashAttention 2/3; with the default SDPA attention packed conversations could attend to each other, so `sft.py` refuses `--packing` unless you also pass `--attn-implementation flash_attention_2`. Padding wastes some compute, which the estimate below includes |
| Loss | **Assistant tokens only** (`--assistant-only-loss`, default on) | Training on user turns teaches the model to write questions too, and wastes 25% of the gradient on text you never want it to produce |

### The box (prices checked 2026-11-13)

The arithmetic is the same $6ND$ as for pretraining: $6 \times 337\text{M params} \times 334\text{M tokens} \times 2\text{ epochs} \approx 1.35 \times 10^{18}$ FLOPs. A Hugging Face Trainer without `torch.compile`, with padded batches, reaches roughly 15–25% MFU on an H100 (ESTIMATE; you measure the real number in the first minutes of 09.2).

| GPU (RunPod) | $/h Community / Secure | SFT, 2 epochs (ESTIMATE) | SFT cost | Notes |
|---|---|---|---|---|
| **1× H100 SXM** | $2.69 / $4.39 | ~2–3 h | **~$5–13** | Your `mygpt-gpu` template from 04.2; DPO (10.1) and the evals (10.2) follow on the same box |
| 1× H100 PCIe | $1.99 / $2.89 | ~2.5–4 h | ~$5–12 | Fine too; ~25% slower |
| 1× RTX 4090 (24 GB) | $0.34 / $0.89 | ~12–18 h | ~$4–16 | Fits a 350M full fine-tune, but a day of babysitting for no saving |

Recommendation: **1× H100 SXM from your template**, the same choice as the plan's "SFT + DPO ~6 GPU-h, $25" line. Multi-GPU is not worth it here: a 2-hour job does not justify an 8× node's setup time.

> [!NOTE]
> Prices drift. Before 09.2, re-check the H100 SXM price on the RunPod deploy page and put the price you will actually pay into `budget.md`.

## Do this

### 1. Get the base model onto your laptop (5 min)

```bash
cd ~/mygpt && . .venv/bin/activate          # Windows: .venv\Scripts\activate
set -a; . ~/.config/mygpt/r2.env; set +a      # RCLONE_CONFIG_R2_* and MYGPT_REMOTE
bash scripts/sync.sh pull-hf my-gpt-350m-base hf/my-gpt-350m-base
cat hf/my-gpt-350m-base/mygpt_export.json
```

`mygpt_export.json` shows the architecture (`LlamaForCausalLM` or `Qwen3ForCausalLM`), `params` (~337M) and the `max_abs_logit_diff_fp32` the export checked. The folder is ~1.35 GB in fp32.

### 2. Install the post-training extras and the lab files (5 min)

Download the [Module 9 lab files](../../labs/module-09/), copy `sft_data_stats.py` and `probe_chat.py` into `tools/`, then:

```bash
pip install -e ".[post]"
git add tools/ && git commit -m "Module 9 tools"
python -c "import trl, transformers; print(trl.__version__, transformers.__version__)"   # 1.17.x 5.22.x
```

### 3. Look at your chat template, token by token (10 min)

The export wrote a plain-text template into the tokenizer: role tags are ordinary text, so no tokens were added and the embedding matrix never changed after pretraining. Render a conversation and the loss mask TRL will use:

```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("hf/my-gpt-350m-base")
msgs = [{"role": "system", "content": "Be brief."},
        {"role": "user", "content": "What is 2+2?"}, {"role": "assistant", "content": "4."},
        {"role": "user", "content": "And 3+3?"}, {"role": "assistant", "content": "6."}]
print(repr(tok.apply_chat_template(msgs, tokenize=False)))
enc = tok.apply_chat_template(msgs, return_dict=True, return_assistant_tokens_mask=True)
print("".join(f"[{tok.decode([i])}]" if m else tok.decode([i])
              for i, m in zip(enc["input_ids"], enc["assistant_masks"])))
print(tok.apply_chat_template(msgs[:2], tokenize=False, add_generation_prompt=True))
```

Real output (2026-11-13, a model exported with the same template):

```text
'<|system|>\nBe brief.\n<|user|>\nWhat is 2+2?\n<|assistant|>\n4.<|endoftext|>\n<|user|>\nAnd 3+3?\n<|assistant|>\n6.<|endoftext|>\n'
<|system|>
Be brief.
<|user|>
What is 2+2?
<|assistant|>
[4][.][<|endoftext|>]
<|user|>
And 3+3?
<|assistant|>
[6][.][<|endoftext|>]
```

Read it carefully, because three things in it decide whether 09.2 works:

- **Bracketed tokens are the only ones in the loss.** The template wraps each assistant answer in `{% generation %}…{% endgeneration %}`; with `assistant_only_loss=True`, TRL turns that into the mask above. User and system turns, and the role tags, are context only.
- **`<|endoftext|>` is inside the brackets.** The model is trained to emit it after every answer. That is how it learns to stop; `chat.py` stops generation on it. TRL warns if a template leaves the end-of-turn token out of the mask; yours does not.
- **The last line prints the generation prompt** `<|user|>\n…\n<|assistant|>\n`. At inference the model continues after `<|assistant|>\n`, exactly where training put the answers.

With your own 32k tokenizer, `<|user|>` is several ordinary tokens (something like `<|`, `user`, `|>`), and `<|endoftext|>` is the single special id 0 from 03.1. Check that the EOT appears as one bracketed token in your output.

### 4. Measure the data through your tokenizer (15 min)

```bash
mkdir -p reports
python tools/sft_data_stats.py --model hf/my-gpt-350m-base --dataset HuggingFaceTB/smol-smoltalk \
  --max-length 1024 --n 3000 --price-per-gpu-hour 2.69 --json reports/sft_data_stats.json
```

If your model's context is 2,048, use `--max-length 2048`. The script streams 3,000 conversations (no full download), renders each with your template, and prints the table you saw above for your tokenizer plus an ESTIMATE line such as:

```text
ESTIMATE: 2 epoch(s) ~ 1.90 GPU-h on 1x H100 at MFU 20% -> ~$5.11 at $2.69/GPU-h
```

That line was produced with the GPT-2 tokenizer and `--params 337e6`; your number will be within a few percent.

### 5. Read the data you are about to train on (10 min)

Statistics do not tell you what the model will learn to sound like. Read ten random conversations:

```python
from datasets import load_dataset
ds = load_dataset("HuggingFaceTB/smol-smoltalk", split="test")
for r in ds.shuffle(seed=1).select(range(10)):
    for m in r["messages"]:
        print(f"--- {m['role']}\n{m['content'][:600]}")
    print("=" * 80)
```

Note what you see: answer length, tone, how often answers use lists, how many conversations have a system message, whether anything is something you would not want to publish. These notes go into the data card.

### 6. Write the data decision (10 min)

Create `reports/sft_data.md`:

```markdown
# SFT data decision (09.1)

- Dataset: HuggingFaceTB/smol-smoltalk, revision <commit sha from the dataset page>, Apache-2.0
- Split: train (460,341 conversations), eval on the first 2,000 of `test`
- Format: `messages`; chat template from mygpt.export_hf (<|system|>/<|user|>/<|assistant|> text tags, EOT ends answers)
- Loss: assistant tokens only (incl. EOT); packing off
- max_length: 1024 (model block_size); truncated: __%; dropped: __%; trained share: __%
- Epochs: 2; tokens per epoch after truncation: ___M
- Estimate: ___ GPU-h on 1x H100 SXM at assumed MFU 20% -> $___ (see sft_data_stats.json)
- What the data looks like (10 samples read): ...
- Alternatives considered: smoltalk (all), Tulu 3 SFT mixture, OpenHermes-2.5 -> rejected because ...
```

Record the dataset revision: on the dataset page, **Files and versions** shows the latest commit hash. Without a pin, `sft.py` downloads whatever version is current on the day you train; in 09.2 you pass this hash as `--dataset-revision`, so the run trains on exactly the data you measured here.

### 7. Commit and add the budget line (2 min)

```bash
git add reports/sft_data.md reports/sft_data_stats.json && git commit -m "SFT data decision" && git push
```

## How you know it worked

- The rendered template matches the pattern above exactly: `<|user|>\n…\n<|assistant|>\n` and every answer ends with `<|endoftext|>` followed by a newline.
- In the mask printout, only answer text and the EOT are bracketed; the role tags and user text are not. The EOT is a single token.
- `sft_data_stats.py` reports a mean of roughly 900–1,050 tokens per conversation, 35–50% truncated at 1,024 (under 3% at 2,048), 1–3% dropped, and a trained share around 75%. Wildly different numbers mean a different template or tokenizer than you think (see below).
- The ESTIMATE line is in the range of 1.5–3 GPU-h for 2 epochs at 1,024 tokens.
- `reports/sft_data.md` names the dataset revision, its license and every number above.

## When it goes wrong

| Symptom | Cause | Fix |
|---|---|---|
| `has no chat template; export it with mygpt.export_hf` | The model folder came from an older export, or you pointed at a checkpoint folder | Re-export with `python -m mygpt.export_hf` (it always writes the template) and push it to `hf/` again |
| Nothing is bracketed in the mask printout | The template lost its `{% generation %}` markers (for example you passed `--chat-template` with your own file) | Use the exported template; TRL would otherwise patch in a generic one or train on nothing |
| `<\|endoftext\|>` shows as several tokens | The tokenizer does not know it as a special token (wrong `tokenizer.json`, or a GPT-2-tokenizer folder for a 32k model) | Check `tok.eos_token_id`: 0 for your BPE, 50256 for GPT-2; re-export with `--tokenizer tokenizers/bpe32k` |
| Mean tokens per conversation far above ~1,000 | You measured `smoltalk` (all) or `smoltalk2` instead of `smol-smoltalk`, or the tokenizer compresses badly | Check `--dataset`; compare bytes/token with 03.1 |
| The stats script is slow or stalls at the start | First streaming request resolves the dataset files | Wait a minute; set `HF_TOKEN` for higher Hub rate limits |
| `datasets` fills your disk | You ran `load_dataset` without `streaming` on several large mixes | Delete `~/.cache/huggingface/datasets/<name>`; only `sft.py` on the GPU box needs the full download |

## Artifact

- `reports/sft_data.md` (dataset, revision, license, max length, epochs, measured statistics, estimate, alternatives) and `reports/sft_data_stats.json`, committed.
- `tools/sft_data_stats.py` and `tools/probe_chat.py` committed for 09.2.
- `budget.md` line:

  ```text
  | 2026-11-13 | 09.1 | SFT data choice and stats on laptop (CPU), datasets free | $0 | $0 |
  ```
