---
# Model card template (lesson 11.1). Copy to release/my-gpt-350m/README.md, fill every <...> and ____,
# delete these comment lines, then run: python tools/check_release.py release/my-gpt-350m
license: apache-2.0
library_name: transformers
pipeline_tag: text-generation
language:
  - en
tags:
  - from-scratch
  - gpt
  - dpo
datasets:
  - HuggingFaceFW/fineweb-edu
  - HuggingFaceTB/smol-smoltalk
  - trl-lib/ultrafeedback_binarized
base_model: <your-username>/my-gpt-350m-base     # remove this line in the base model's own card
co2_eq_emissions:
  emissions: ____            # grams CO2e, ESTIMATE (see Compute section)
  source: "Estimate: GPU-hours x 0.7 kW x PUE 1.2 x grid intensity"
  training_type: pre-training
  geographical_location: "<provider region, e.g. RunPod EU-RO-1>"
  hardware_used: "8x NVIDIA H100 SXM 80GB (pretraining), 1x H100 (post-training)"
model-index:
  - name: my-gpt-350m
    results:
      - task:
          type: text-generation
        dataset:
          name: HellaSwag
          type: hellaswag
        metrics:
          - name: acc_norm (0-shot)
            type: acc_norm
            value: ____
        source:
          name: lm-eval-harness 0.4.13, own run (see Evaluation)
          url: https://github.com/<your-username>/<your-repo>/blob/v1.0/reports/eval_report.md
---

# my-gpt-350m

A ~<337>M-parameter decoder-only language model trained **from scratch** by <your name> as a
learning project: own 32,768-token BPE tokenizer, pretraining on ~<20>B tokens of FineWeb-Edu,
then supervised fine-tuning (SFT) and Direct Preference Optimization (DPO). It is a small research
model, comparable in size to GPT-2 medium, and **not** a capable assistant.

| | |
|---|---|
| Revisions | `main` / tag `v1.0` = DPO chat model; branch `sft` = SFT model; base model: [`<your-username>/my-gpt-350m-base`](https://huggingface.co/<your-username>/my-gpt-350m-base) |
| Architecture | `<Qwen3ForCausalLM>`: <24> layers, d_model <1024>, <16> heads, RoPE, RMSNorm, SwiGLU, QK-norm, no biases, tied embeddings, context <1,024> tokens |
| Parameters | <337>M (embeddings <33.6>M) |
| Weights | `model.safetensors`, <fp32>, <1.35> GB |
| Code | <https://github.com/your-username/your-repo> at tag `v1.0` |
| Technical report | <link> |
| Demo | <https://huggingface.co/spaces/your-username/my-gpt-350m-chat> |

## How to use

```python
from transformers import pipeline

chat = pipeline("text-generation", model="<your-username>/my-gpt-350m", revision="v1.0")
messages = [{"role": "user", "content": "Explain photosynthesis in two sentences."}]
print(chat(messages, max_new_tokens=160, do_sample=True, temperature=0.7, top_p=0.9)[0]["generated_text"][-1]["content"])
```

The chat template uses plain-text role tags (`<|user|>`, `<|assistant|>`); each answer ends with
`<|endoftext|>`. Use `revision="sft"` for the SFT model.

## Intended use

- Education and research: studying how a small model trained on a known budget behaves, comparing
  training recipes, testing evaluation or alignment code on a cheap model.
- **Out of scope:** any decision about people, medical/legal/financial advice, factual lookup,
  production chat, content for children, use as a safety filter.

## Training data

| Stage | Data | Revision | License | Amount used |
|---|---|---|---|---|
| Tokenizer | FineWeb-Edu `sample/100BT` | `<sha>` | ODC-By 1.0 | <2M> documents |
| Pretraining | FineWeb-Edu `sample/100BT`, first <30> files, decontaminated | `<sha>` | ODC-By 1.0 | <20.0>B tokens (own tokenizer) |
| SFT | HuggingFaceTB/smol-smoltalk | `<sha>` | Apache-2.0 | <n> conversations x <2> epochs |
| DPO | trl-lib/ultrafeedback_binarized (UltraFeedback, MIT) | `<sha>` | MIT | <n> pairs x <1> epoch |

**Attribution.** Pretrained on FineWeb-Edu by Hugging Face
(<https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu>), made available under the Open Data
Commons Attribution License (ODC-By) v1.0; its use is also subject to the Common Crawl Terms of Use
(<https://commoncrawl.org/terms-of-use>). Data card: <link to DATA_CARD.md>.

Decontamination: 13-gram overlap with HellaSwag, ARC-Easy/Challenge and PIQA removed before
training (<n> documents dropped; report <link>).

## Training procedure

| | Pretraining | SFT | DPO |
|---|---|---|---|
| Hardware | 8x H100 SXM 80GB, <provider> | 1x H100 | 1x H100 |
| Optimizer | <Muon (blocks) + AdamW> | AdamW | AdamW |
| LR / schedule | <0.02 Muon, 6e-4 AdamW / WSD, 20% decay> | <lr> cosine | <lr>, beta <0.5> |
| Batch | <524,288> tokens | <...> | <...> |
| Precision | bf16 autocast | fp32 master + bf16 autocast | same |
| Wall time | <5.7> h | <h> | <h> |

The recipe was chosen by ablations against a GPT-2 124M baseline (table in the technical report).

## Evaluation

0-shot, lm-eval-harness 0.4.13, log-likelihood scoring, no chat template, same settings for all rows.

| Model | Params | Train tokens | HellaSwag acc_norm | ARC-E acc_norm | ARC-C acc_norm | PIQA acc_norm | Val bpb (common val) |
|---|---|---|---|---|---|---|---|
| OpenAI GPT-2 (124M) | 124M | ~? (WebText) | ____ | ____ | ____ | ____ | ____ |
| OpenAI GPT-2 medium (355M) | 355M | ~? (WebText) | ____ | ____ | ____ | ____ | ____ |
| My baseline (GPT-2 124M repro) | 124M | 10B | ____ | ____ | ____ | ____ | ____ |
| **my-gpt-350m-base** | <337M> | <20B> | ____ | ____ | ____ | ____ | ____ |
| my-gpt-350m (SFT) | | | ____ | ____ | ____ | ____ | |
| my-gpt-350m (DPO, this repo) | | | ____ | ____ | ____ | ____ | |

Standard errors: HellaSwag ~0.5, ARC-E ~1.0, ARC-C ~1.4, PIQA ~1.2 points.

**Pairwise win rate (DPO vs SFT):** ____% (95% CI ____–____%), 200 `no_robots` test prompts, both
answer orders, judge `claude-opus-5-5`; position-pick rate ____%, mean answer length SFT ____ /
DPO ____ words. Details: <link to alignment_results.md>.

## Limitations and bias

- Small model, short context (<1,024> tokens): invents facts, loses track in long conversations,
  weak at arithmetic, code and multi-step reasoning. <Two real failure examples from your probes.>
- English only; trained on educational web text, so it knows little about dialogue, fiction,
  recent events (data up to <dump date>) and non-English topics.
- Web data carries the social biases and stereotypes of its sources; FineWeb-Edu's classifier
  filters for educational style, not for fairness or truth. No bias evaluation was run beyond
  <what you did>.
- SFT/DPO data was largely generated or rated by larger LLMs; the model inherits their style and
  some of their mistakes. Alignment is light: it can still produce harmful or offensive text.
- Benchmark scores are near chance for some tasks at this size; differences under ~2 standard
  errors are noise.

## Compute and environmental impact

| | GPU-hours | Cost (actual) |
|---|---|---|
| Pretraining main run | <45.6> H100-h | $<201> |
| Baseline + ablations | <...> | $<...> |
| SFT + DPO + evaluation | <...> | $<...> |
| **Whole project** | <...> | $<...> (ledger in the technical report) |

Energy and CO2 (ESTIMATE): <GPU-hours> H100-h x 0.7 kW (H100 SXM TDP) x PUE 1.2 = <kWh> kWh;
x <grid intensity> kg CO2e/kWh (<region>, Electricity Maps, <date>) = <kg> kg CO2e. An upper-bound
style estimate: GPUs rarely draw full TDP the whole time, and CPU, network and storage are not
counted.

## License

Weights and code: Apache-2.0. Training data keeps its own terms (FineWeb-Edu ODC-By 1.0 and the
Common Crawl Terms of Use; smol-smoltalk Apache-2.0; UltraFeedback MIT). You are responsible for
complying with those when you use or redistribute the data.

## Citation

```bibtex
@misc{<yourkey>2026mygpt,
  title  = {my-gpt-350m: a GPT-2-class model trained from scratch on a $500 budget},
  author = {<Your Name>},
  year   = {2026},
  url    = {https://huggingface.co/<your-username>/my-gpt-350m}
}
```
