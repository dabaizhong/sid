# LIGER / Beauty — local reproduction, seed 42

Download [`best.pt`](best.pt) (36,420,469 bytes, approximately 34.7 MiB).

- 9,089,536 parameters; official T5 encoder/decoder configuration (6 + 6 layers).
- Selected step **102,580**, epoch **230**, full warm validation Dense NDCG@10
  **0.06028494984**. Training stopped by official patience at step 125,326.
- Original configured limit: 200,000 steps, batch 256, lr .0003, 10,000 warmup
  steps, uninterrupted cosine schedule. Complete configuration is in
  `training_manifest.json` and inside the checkpoint.
- Assets: this repository's `LIGER/Beauty/sid.npy` and
  `embeddings_standardized.npy` (do not standardize a second time).
- Apply SID vocabulary offsets `[1, 257, 513, 769]` to raw SID rows.
- This is the baseline before the incremental alignment experiments.

## Full test results

The checkpoint was chosen using validation, not these test metrics.

| Slice | Retrieval | Recall@10 | NDCG@10 |
|---|---|---:|---:|
| In-set (22,201) | Generation only | 0.07166344 | 0.03630295 |
| In-set | Dense full catalog | 0.08810414 | 0.04517188 |
| In-set | Hybrid K20 | 0.07823972 | 0.04128175 |
| In-set | Hybrid K80 | 0.08414035 | 0.04375660 |
| Cold-start (162) | Hybrid K20 | 0.08641975 | 0.03116666 |

Evaluation uses the upstream unconstrained HF Beam100, then the indicated K
generated candidates and the official cold-item union for dense reranking.
`test_metrics.json` contains all K values and metrics; `paper_comparison.json`
contains the paper comparisons. In-set reproduction reached the target;
cold-start results are below the paper's K20 values.

## Loading

Install PyTorch and Transformers, and clone the upstream code:

```bash
git clone https://github.com/facebookresearch/liger.git
python load_checkpoint.py --upstream /path/to/liger
```

The helper loads the original upstream `src.tiger.TIGER`, reconstructs its
configuration from the checkpoint, and loads the state dict strictly. It does
not require our local training project. Tested with torch 2.7.1 and transformers
4.57.6. The model is returned on CPU in evaluation mode; move to CUDA as needed.
Use the upstream input/evaluation functions with the matching SID and text
tables for inference.

Checkpoint keys: `model`, `step`, `epoch`, `best`, `manifest`. `model` is the
unwrapped upstream TIGER state dict. This best checkpoint contains model weights
and training metadata, not optimizer state for exact training resumption.

SHA-256 (`best.pt`):
`efc50b5cdc213f13b303b2b6e6abba563acdb8567766d8fd33b117881f28f721`

Upstream code provenance and licenses are preserved in this repository's
`sources.json`, `upstream_licenses/`, and `training_manifest.json`.
