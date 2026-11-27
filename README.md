# SID tables for LETTER, ETEGRec, RPG and LIGER

Ready-to-load semantic item IDs (SIDs) and item mappings for the authors' datasets.

| Method | Dataset | Items | SID length | Items sharing a complete SID |
|---|---|---:|---:|---:|
| LETTER | Beauty | 12,101 | 4 | 24 |
| LETTER | Instruments | 9,922 | 4 | 48 |
| LETTER | Yelp | 20,033 | 4 | 1247 |
| ETEGRec | instrument | 24,587 | 4 | 0 |
| ETEGRec | scientific | 25,848 | 4 | 0 |
| ETEGRec | game | 25,612 | 4 | 0 |
| RPG | Beauty | 12,101 | 32 | 6 |
| LIGER | Beauty | 12,101 | 4 | 0 |

## Load a table

```python
import json
import numpy as np
from pathlib import Path

root = Path("LIGER/Beauty")
sid = np.load(root / "sid.npy", allow_pickle=False)
item_ids = json.loads((root / "item_ids.json").read_text())
# Row r in sid belongs to item_ids[r].
item_to_sid = dict(zip(item_ids, sid.tolist()))
print(item_ids[0], sid[0])
```

Each dataset folder contains:

- `sid.npy`: integer array `[number_of_items, SID_length]`.
- `item_ids.json`: item identifiers in exactly the same order as the array rows.
- `internal_ids.json`: the upstream internal identifier corresponding to each row.
- `item_to_sid.json`: direct item-to-SID lookup.
- `sid_to_items.json`: inverse lookup, preserving **all** items when codes collide.
- `manifest.json`: provenance, tokenizer settings and checksums.

LETTER also includes `author_tokens.json` with the author's original token strings.
These are per-method catalogs: matching numeric internal IDs across methods does not imply matching items.
Use ASINs where available. LETTER retains author internal item IDs.

## How the tables were obtained

- **LETTER:** author-published four-token SID files, converted to integer arrays without changing codes. Original collisions are preserved.
- **ETEGRec:** initial SID snapshot exported from the author's pretrained RQ-VAE and supplied item embeddings. The fourth token disambiguates collisions. Joint ETEGRec training subsequently updates the tokenizer; these are not final jointly trained recommendation checkpoints.
- **RPG / Beauty:** locally generated using Sentence-T5-base, PCA to 128 dimensions with whitening, and the upstream OPQ32 / IVF1 / PQ32x8 tokenizer. The table contains 32 raw code values; this is the base-text-encoder variant. Original OPQ collisions are preserved.
- **LIGER / Beauty:** locally generated using Sentence-T5-XXL, StandardScaler, and the upstream three-level RQ-VAE with 256 entries per level, trained for 8,000 epochs. A fourth collision suffix makes the complete SID unique. Raw code values have **no** model-vocabulary offsets; the upstream LIGER loader applies offsets `[1, 257, 513, 769]`.

For LIGER, the large text encoder was evaluated layer by layer to fit a 6GB GPU. Final vectors were checked against ordinary Hugging Face forward outputs on reference items (minimum cosine similarity 0.99999976).

## Beauty 768-dimensional text vectors

`LIGER/Beauty/` also contains the cached Sentence-T5-XXL vectors generated on the local RTX 3060:

| File | Shape / dtype | Use |
|---|---|---|
| `embeddings.npy` | `(12101, 768)`, float32 | Original final L2-normalized Sentence-T5-XXL text vectors |
| `embeddings_standardized.npy` | `(12101, 768)`, float32 | Standardized vectors consumed by our LIGER / RQ-VAE training |
| `standardization.npz` | mean, scale, var | Numeric StandardScaler parameters; no pickle loading required |
| `encoder_check.json` | JSON | Reference-forward consistency check |

Both vector files have **exactly the same row order** as `LIGER/Beauty/sid.npy`, `item_ids.json` and `internal_ids.json`. Row 0 is the first real item, not padding. Each `.npy` vector file is 37,174,400 bytes (about 35.5 MiB). Use the standardized file for the matching LIGER run; do not standardize it again. To join with another method's SID table, match item identifiers rather than assuming identical row order.

The encoder was evaluated in mixed precision and its output cached as float32. The original normalization is preserved byte-for-byte; vector norms can differ from 1 by FP16 rounding (less than 0.001).

```python
root = Path("LIGER/Beauty")
text_vectors = np.load(root / "embeddings.npy", allow_pickle=False)
liger_vectors = np.load(root / "embeddings_standardized.npy", allow_pickle=False)
# Optional PyTorch conversion: torch.from_numpy(liger_vectors)
```

## Trained LIGER Beauty checkpoint

The locally reproduced **9.09M-parameter LIGER, seed 42** is available at
[`LIGER/Beauty/checkpoints/seed42/`](LIGER/Beauty/checkpoints/seed42/).
The selected checkpoint is step **102,580**, epoch **230**, chosen by full
validation Dense NDCG@10. It uses this repository's Beauty SIDs and standardized
768-dimensional vectors. In-set hybrid K20 Recall@10 / NDCG@10 are
**0.07823972 / 0.04128175**. Full metrics, configuration, loading code and the
cold-start results are included in that folder. This is a local reproduction,
not an author-released checkpoint.

## Sources and attribution

- LETTER: https://github.com/HonghuiBao2000/LETTER
- ETEGRec: https://github.com/RUCAIBox/ETEGRec
- RPG: https://github.com/facebookresearch/RPG_KDD2025
- LIGER: https://github.com/facebookresearch/liger

`sources.json` records the upstream code revisions used. This is a collection of author-published and locally generated SID assets, not an official repository of those projects. Upstream code licenses available in the downloaded repositories are preserved under `upstream_licenses/`. Original dataset and asset rights remain with their respective owners; this collection does not relicense them. Please cite the corresponding papers and datasets when using these tables.

## Verification

```bash
pip install numpy
python verify.py
```

`SHA256SUMS.json` provides checksums for the packaged files. This repository contains SID assets, cached Beauty text embeddings, the locally trained LIGER Beauty checkpoint and loading utilities; user interaction logs and user identifier mappings are not included.
