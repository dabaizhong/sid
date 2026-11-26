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

`SHA256SUMS.json` provides checksums for the packaged files. This repository contains SID assets and loading utilities; model weights, user interaction logs and user identifier mappings are not included.
