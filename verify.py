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
    print(f"OK {entry['path']}: {len(ids):,} items, {array.shape[1]} tokens")
print('All file hashes, SID rows, item mappings and inverse mappings verified.')
