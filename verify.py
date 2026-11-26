import hashlib
import json
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parent
for relative, expected in json.loads((root / 'SHA256SUMS.json').read_text()).items():
    assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == expected, relative
for entry in json.loads((root / 'catalog.json').read_text()):
    folder = root / entry['path']
    array = np.load(folder / 'sid.npy', allow_pickle=False)
    ids = json.loads((folder / 'item_ids.json').read_text())
    mapping = json.loads((folder / 'item_to_sid.json').read_text())
    inverse = json.loads((folder / 'sid_to_items.json').read_text())
    assert array.shape == (entry['items'], entry['code_length'])
    assert len(set(ids)) == len(ids) == len(array)
    for item, code in zip(ids, array.tolist()):
        assert mapping[item] == code
        assert item in inverse[','.join(map(str, code))]
    assert sum(map(len, inverse.values())) == len(ids)
    assert len(inverse) == entry['unique_codes']
    manifest = json.loads((folder / 'manifest.json').read_text())
    if 'embedding_files' in manifest:
        for name, info in manifest['embedding_files'].items():
            vectors = np.load(folder / name, allow_pickle=False)
            assert list(vectors.shape) == info['shape'] == [len(ids), 768]
            assert str(vectors.dtype) == info['dtype'] == 'float32'
            assert np.isfinite(vectors).all()
            assert hashlib.sha256((folder / name).read_bytes()).hexdigest() == info['sha256']
        original = np.load(folder / 'embeddings.npy', allow_pickle=False)
        standardized = np.load(folder / 'embeddings_standardized.npy', allow_pickle=False)
        # Encoder normalization was evaluated in mixed precision, then cached as float32.
        np.testing.assert_allclose(np.linalg.norm(original, axis=1), 1., atol=1e-3)
        with np.load(folder / 'standardization.npz', allow_pickle=False) as params:
            reproduced = original.copy()
            reproduced -= params['mean']
            reproduced /= params['scale']
        np.testing.assert_allclose(standardized, reproduced, atol=2e-6, rtol=2e-6)
        print(f"OK {entry['path']}: raw/standardized 768-d vectors and scaler")
    print(f"OK {entry['path']}: {len(ids):,} items, {array.shape[1]} tokens")
print('All file hashes, SID rows, item mappings and inverse mappings verified.')
