# Data card — <your project name> pretraining data

Fill every field. Keep this file in your project repo (`DATA_CARD.md`) and a copy at the root of
your bucket. It is cited by your model card in Module 11.

## Summary

- **Dataset:** FineWeb-Edu (`HuggingFaceFW/fineweb-edu`), English educational web pages filtered
  from Common Crawl by an educational-quality classifier (score >= 3 of 5).
- **Used for:** pretraining. Baseline (GPT-2 124M): `sample/10BT`. Main run (~350M): first
  `<N>` files of `sample/100BT`.
- **Dataset revision (commit sha):** `<sha from manifest>`
- **Downloaded on:** `<YYYY-MM-DD>` from `<machine / provider / region>`

## License and terms

- **License:** ODC-By 1.0 (Open Data Commons Attribution). Attribution required when you share
  the data or derived databases.
- **Upstream terms:** use is also subject to the Common Crawl Terms of Use.
- **What this allows for this project:** training and publishing model weights; republishing the
  raw text would need the same attribution.
- **Attribution text used in the model card:** `<e.g. "Trained on FineWeb-Edu (Hugging Face,
  ODC-By 1.0), https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu">`

## What we have

| Subset | Files | Bytes (parquet) | Est. GPT-2 tokens | Manifest |
|---|---|---|---|---|
| sample/10BT | 14 | `<bytes>` | ~10B | `manifests/sample-10BT.json` |
| sample/100BT (first N files) | `<N>` | `<bytes>` | `<est>` | `manifests/sample-100BT-<N>.json` |

- **Integrity check:** sha256 of every file matched the Hub at the revision above (`problems: 0`).
- **Columns:** `text, id, dump, url, file_path, language, language_score, token_count, score,
  int_score`.

## Where it lives

- **Bucket:** `<r2:mygpt>`; raw files under `raw/fineweb-edu/<subset>/`.
- **Storage provider and price:** `<Cloudflare R2, $0.015/GB-month, checked YYYY-MM-DD>`

## Known properties and risks

- English only; skewed toward educational/explanatory writing (by design), so less dialogue,
  fiction and code than a general web mix.
- `sample/10BT` was drawn from `sample/100BT`, so the two overlap; evaluation splits are made in
  Module 3, not here.
- May contain benchmark text (HellaSwag, ARC, PIQA questions are web-derived): checked by the
  decontamination step in lesson 03.3. Result: `<link to report>`.
- May contain personal data and offensive text that the upstream filters missed.

## Changes we made

- `<none at this stage | list filters, dropped files, dedup>`
