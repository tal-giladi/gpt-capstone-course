# Module 9 lab files — instruction tuning (SFT)

Helpers for lessons 09.1 and 09.2. Copy them into the `tools/` folder of your project repo and
commit them, so every rented box gets them with `git clone`. The code they drive
(`scripts/sft.py`, `scripts/chat.py`, `mygpt/export_hf.py`) is in `labs/common/`.

| File | Used in | What it does |
|---|---|---|
| `sft_data_stats.py` | 09.1 (also 10.1 with `--kind dpo`) | Streams a sample of a chat (or preference) dataset, renders it with **your** tokenizer and chat template, and reports tokens per conversation, the share cut by `--max-length`, the share of trained (assistant) tokens, and conversations TRL will drop because truncation removed every assistant token. Extrapolates to the whole split and prints an ESTIMATE of GPU-hours and dollars from 6·N·tokens (SFT) or 8·N·tokens (DPO) at an assumed MFU. CPU only. |
| `probe_chat.py` | 09.2, 10.1 | Sends a fixed set of 12 probe prompts (or your own file) to one or more models with greedy decoding and the same settings `judge_winrate.py` uses, and writes a side-by-side markdown report, including whether each model stopped by itself. |

Examples:

```bash
python tools/sft_data_stats.py --model hf/my-gpt-350m-base --dataset HuggingFaceTB/smol-smoltalk --max-length 1024
python tools/sft_data_stats.py --model hf/my-gpt-350m-sft --dataset trl-lib/ultrafeedback_binarized --kind dpo
python tools/probe_chat.py --models hf/my-gpt-350m-base hf/my-gpt-350m-sft --out reports/sft_probe.md
```

Requirements: the project installed with `pip install -e ".[post]"` (transformers, datasets,
TRL). `sft_data_stats.py` downloads only the rows it samples (streaming).

Tested on 2026-11-13 (Windows 11, Python 3.14, CPU, transformers 5.22.2, datasets 5.2.0,
TRL 1.17.1) against a tiny Qwen3-architecture model exported with `mygpt.export_hf` and the
GPT-2 tokenizer: `sft_data_stats.py` on 3,000 rows of `HuggingFaceTB/smol-smoltalk` and of
`trl-lib/ultrafeedback_binarized`, `probe_chat.py` on a base and an SFT model.
