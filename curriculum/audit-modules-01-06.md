# Audit — Modules 1–6 (resource alternatives and artifact chain)

Date: 2026-11-13. Scope: `lessons/module-01..06`, `labs/module-01..06`, quizzes of those modules.
Prices checked 2026-11-13 on the provider pages cited in each lesson's `sources`.
Dry-run after the changes: `npm run check-course` → 0 problems.

## Cross-cutting changes

- **Storage alternatives without rewriting commands.** Every lesson uses `r2:mygpt/...`. Instead of
  changing those, 02.2 and `labs/module-02/r2.env.example` now show how to keep the remote *name*
  `r2` as an rclone **alias** over a second remote `store` (B2, Hetzner Object Storage, Wasabi,
  AWS S3, GCS, HF Storage Bucket). All later paths, `MYGPT_REMOTE` and `ckpt_remote` stay valid.
  Notes the global bucket-name namespace on B2/S3/GCS/Wasabi.
- **Stop/terminate commands per provider** (04.2): RunPod `runpodctl pod stop`, Vast.ai
  `vastai destroy instance $CONTAINER_ID --api-key $CONTAINER_API_KEY`, Lambda Cloud API
  `instance-operations/terminate` (cloud.lambda.com). Plus an optional, cheap test of the fallback's
  stop path (requested by the Modules 7–11 audit, so 07.1 can rely on it).
- Lambda doc URLs moved from docs.lambda.ai to docs.lambda.com (01.1, 01.2, 04.1, 04.2).
- Experiment tracking: W&B offline, no-tracker (`log.jsonl`), Trackio via `train.tracker=trackio`
  (being added to `labs/common` by the other session), Comet; Neptune marked as shut down (2026-03-06).
- Judge API (01.2): Gemini, OpenRouter/Groq (`gpt-oss-120b`), self-hosted vLLM, matching Module 10.

## Per lesson

| Lesson | Resources needing an account or money | Alternatives added (recommended + ≥2) | Artifact → consumer | Other changes |
|---|---|---|---|---|
| 01.1 | none (planning) | New table "Every paid resource has a fallback": 8×H100, 1×H100, CPU box, storage, tracking, code hosting, judge API, password manager | `budget.md` → 01.3 ✓ | sources +5 |
| 01.2 | GitHub, HF, W&B, Cloudflare, RunPod, Hetzner, Vast.ai, Lambda, LLM API, password manager | New table: GitLab Free / private HF repo; W&B offline / Trackio / Comet; B2 / Hetzner OS; Vast.ai / Lambda / Verda; RunPod CPU / DigitalOcean; Gemini / OpenRouter / Groq / vLLM; Bitwarden / 1Password / KeePassXC | `secrets.env` → 01.3, 02.x ✓ | sources +15; Lambda URL |
| 01.3 | code hosting | Code hosting row: GitHub / GitLab Free / private HF repo, with the GitLab push command | repo `<you>/mygpt` → every box `git clone` ✓ | — |
| 02.1 | HF (free) | "How to read a sample": datasets streaming / single parquet via `hf download` / dataset viewer | `DATA_CARD.md` draft → 02.3 ✓ | — |
| 02.2 | object storage | +Hetzner Object Storage, Wasabi, GCS rows (had R2, B2, S3, HF, RunPod NV, Nebius); "If R2 is not possible" pick table; "Using another provider" with per-provider rclone settings | bucket + `r2.env` + layout → 02.3, 03.x, 04.2 template ✓ | `r2.env.example`: alt provider blocks; its comment said `rclone lsd r2:` (fails with bucket-scoped tokens) → `rclone lsf r2:mygpt` |
| 02.3 | CPU box, HF | +DigitalOcean CPU-Optimized row; download-method table (`download_fineweb.py` / `hf download` / HF bucket server-side copy); delete commands for RunPod/DigitalOcean | `raw/fineweb-edu/...`, `manifests/` → 03.1, 03.2 ✓ | — |
| 03.1 | CPU box | +DigitalOcean row; tokenizer-library table (HF tokenizers / SentencePiece / tiktoken); how to start on RunPod CPU or DigitalOcean | `tokenizer/`, `tokenizers/bpe32k` → 03.2 ✓ | — |
| 03.2 | CPU box | +DigitalOcean 32 vCPU row; cost line for fallbacks; delete commands | `tok/gpt2-10BT/`, `tok/bpe32k-100BT-30/` → 03.3, 04.3, 05.x, 06.x ✓ | — |
| 03.3 | CPU box | New box table: CPX62 / CPX42 / RunPod CPU or DigitalOcean | `tok/common-val-*`, `TOKEN_REPORT.md` → 08.2 ✓ | `--workers 15` → `$(( $(nproc) - 1 ))` (CPX42 has 8 vCPU) |
| 04.1 | GPU clouds, secrets | +Verda (ex-DataCrunch), TensorDock, Paperspace rows; Colab/Kaggle marked unusable; Hyperbolic now from its own marketplace page; secrets on non-RunPod providers (`scp` env files) | price table + RunPod account → 04.2 ✓ | sources +5 |
| 04.2 | 1×H100 | Provider row (RunPod / Vast.ai / Lambda / Verda); "On another provider" table (image, stop command, finish); optional fallback stop-path test | `mygpt-gpu` template, `env/` record → 04.3, 05.1 ✓ | sources +3 |
| 04.3 | 1×H100 | Box table: RunPod Community / Vast.ai / Lambda | bench files + revised budget → 05.3, 06.4 ✓ | **Fixed:** shard pull used placeholders `<baseline-folder>`/`<data_dir ...>`; now `tok/gpt2-10BT` → `data/tok/gpt2-10BT` and `tok/bpe32k-100BT-30` → `data/tok/bpe32k-100BT-30` (incl. `tokenizer.json`); "200M tokens each" → 100M |
| 05.1 | 2-GPU box, 8× node, W&B | +Vast.ai 2×4090 row; tracking table (W&B / offline / none / Trackio); W&B key on non-RunPod boxes | `launch/baseline_8xh100.sh`, `mygpt-8gpu` → 05.3, 07.1 ✓ | sources +2 |
| 05.2 | 2-GPU box, bucket | Box table: RunPod 4090 / RunPod A100 / Vast.ai; note that the drill works with any aliased bucket | `reports/drill-05.2.md` → 07.2 (recovery) ✓ | — |
| 05.3 | 8×H100 | "Where to run it" table: RunPod / Lambda / Vast.ai / Nebius with predicted cost and what changes | `ckpt/baseline_124m/`, `results.md` → 06.4, 07.2, 08.1, 08.3 ✓ | Artifact said console log "under reports/" but step 7 copies it to the repo root → now "moved to `reports/baseline_124m.out`" |
| 06.1 | 1×H100 | +Verda spot row (had Vast.ai, H100 PCIe, spot); stop command on other providers | `results/ablations.md` v1 → 06.2 ✓ | **Fixed** broken command: literal `\n` instead of a line continuation in the smoke-queue command |
| 06.2 | 1×H100 | Box table: RunPod / Vast.ai / Verda spot with block cost | `results/ablations.md` v2 → 06.3 ✓ | sources +3 |
| 06.3 | 1×H100 | Same box table | `results/ablations.md` v3 → 06.4 ✓ | sources +3 |
| 06.4 | 8× price, optional 1×H100 | +Nebius row (had RunPod, Lambda, Vast.ai); LR-check box alternatives | `configs/main_350m.yaml` @ `recipe-v1` with `ckpt_remote: r2:mygpt/ckpt/main_350m`, `decisions/0001-*.md` → 07.1 ✓ | sources +2 |

Quizzes: no fact a question depends on changed; none edited. All quiz YAML in scope parses.

## Cross-scope (Modules 7+) — checked, no mismatch

- 07.1 consumes `recipe-v1`, `ckpt_remote: r2:mygpt/ckpt/main_350m`, `wandb_project: my-gpt`,
  `mygpt-8gpu`, `launch/baseline_8xh100.sh` — all produced as named.
- 08.1/08.2 consume `ckpt/baseline_124m/`, `tok/common-val-gpt2/`, `tok/common-val-bpe32k/`,
  `TOKEN_REPORT.md` — all produced as named.
- 11.3 lists the committed artifacts but not `results.md` (05.3 baseline table) or
  `reports/drill-05.2.md`; harmless, the baseline numbers reach it through `EVAL_REPORT.md`.
- The `r2` alias approach means Module 7–11 commands work unchanged on any storage provider;
  Module 7+ RunPod-specific steps (template, `runpodctl`) are the other audit's scope.

## Not verifiable / estimates left marked

- RunPod CPU pod price: not on the public pricing page; kept as "~$0.03–0.04 per vCPU-hour, check the console".
- Google Cloud Storage internet egress (~$0.12/GB): the pricing page did not render; storage $0.020/GB-month from the docs page. Marked "~".
- Colab Pro price ($9.99/100 units) and Kaggle quota: third-party sources only; stated as such.
- Verda 8×H100 and TensorDock 8× offers are not on their public pages (said so).
- Lambda terminate endpoint: path from Lambda's OpenAPI spec, host `cloud.lambda.com` from the current API docs; not executed.
- Hetzner S3-credential menu location and Wasabi region endpoint follow the providers' usual docs; not clicked through.
