# Module 11 lab files — Ship it

Copy the Python files into your project repo's `tools/` folder and commit them. The templates go
where the lessons say. Tested on 2026-11-13 on Windows 11, Python 3.14, torch 2.15.0+cpu,
transformers 5.22.2, gradio 6.32.0, with a tiny exported Qwen3 model (not your real model).

| File | Lesson | What |
|---|---|---|
| `check_release.py` | 11.1 | Checks a model folder before upload: safetensors only, no pickles/trainer state/secrets, config + tokenizer + README with license metadata; `--load` loads it and generates |
| `MODEL_CARD.template.md` | 11.1 | The model card (`README.md` of the Hub repo), with metadata, data attribution, eval tables, limitations, compute and CO2 |
| `cpu_speed.py` | 11.2 | Tokens per second of your model on N CPU threads, to size a CPU host |
| `space/` | 11.2 | The demo Space: `app.py` (Gradio chat, streaming, rate limit, disclaimer; ZeroGPU-ready), `requirements.txt` (pinned), `README.md` (Space YAML header) |
| `ledger_summary.py` | 11.3 | Sums `budget.md`: planned vs actual per module and total, lists rows without an actual |
| `TECH_REPORT.template.md` | 11.3 | Technical report skeleton |

```bash
python tools/check_release.py release/my-gpt-350m --load
python tools/cpu_speed.py --model release/my-gpt-350m --threads 2
MODEL_ID=release/my-gpt-350m python space/app.py          # http://127.0.0.1:7860
python tools/ledger_summary.py budget.md --cap 500
```

`space/app.py` needs `gradio==6.32.0` plus `space/requirements.txt` (`pip install -e ".[demo]"`
in the project repo already has them, with torch 2.15).
