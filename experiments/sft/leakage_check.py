r"""Audit train/evaluation text overlap for RadioReady (no GPU or extra packages).

From Windows PowerShell in the repository root, run:
  .\.venv\Scripts\python.exe radioready_split_leakage_check.py

Required files:
  data/processed/sft/sft_plain.jsonl
  outputs/evaluation/expanded_test/predictions.jsonl
"""
import json
import re
import unicodedata
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

TRAIN = Path('data/processed/sft/sft_plain.jsonl')
EVAL = Path('outputs/evaluation/expanded_test/predictions.jsonl')

for filepath in [TRAIN, EVAL]:
    if not filepath.is_file():
        raise SystemExit(f'Missing {filepath.resolve()}')


def load(path):
    return [json.loads(line) for line in path.open(encoding='utf-8') if line.strip()]


def normalize(text):
    text = unicodedata.normalize('NFKC', text).lower()
    return re.sub(r'\s+', ' ', re.sub(r'[^\w\s]', ' ', text)).strip()

all_sft = load(TRAIN)
eval_rows = load(EVAL)
train_rows = [r for r in all_sft if r.get('split') == 'train'
              and r.get('source') in ('genius', 'kidzbop')]

# sft_plain.prompt contains one instruction line, followed by the original lyrics.
train = [dict(id=r['id'], source=r['source'], text=normalize(r['prompt'].split('\n', 1)[1]))
         for r in train_rows]
test = [dict(id=r['id'], source=r['source'], text=normalize(r['original']))
        for r in eval_rows]

print(f'Training lyric sections: {len(train)}, selected held-out sections: {len(test)}')
print('Training by source:', dict(Counter(r['source'] for r in train)))
print('Test by source:', dict(Counter(r['source'] for r in test)))

train_ids = {r['id'] for r in train}
ids_in_train = [r['id'] for r in test if r['id'] in train_ids]
print(f'\nExact train/test ID collisions: {len(ids_in_train)}', ids_in_train[:12])

train_by_text = defaultdict(list)
for r in train:
    train_by_text[r['text']].append(r['id'])

exact = [(r['id'], train_by_text[r['text']]) for r in test if r['text'] in train_by_text]
print(f'Normalized exact-section matches to train: {len(exact)}')
for item in exact[:12]:
    print('  test ID:', item[0], 'train IDs:', item[1])

# Approximate near matches. These may be legitimate repeated choruses,
# artist-associated duplicate lyrics, or coincidences; inspect individually.
near = []
for tr in train:
    for te in test:
        if tr['source'] != te['source'] or tr['text'] == te['text']:
            continue
        if min(len(tr['text']), len(te['text'])) < 40:
            continue
        # Cheap length-ratio bound avoids many unnecessary comparisons.
        max_possible = 2 * min(len(tr['text']), len(te['text'])) / (len(tr['text']) + len(te['text']))
        if max_possible < 0.92:
            continue
        score = SequenceMatcher(None, tr['text'], te['text'], autojunk=False).ratio()
        if score >= 0.92:
            near.append((score, te['id'], tr['id'], te['source']))
near.sort(reverse=True)
print(f'\nNear-matching train/test section pairs (char similarity >= 0.92): {len(near)}')
for sim, test_id, train_id, source in near[:20]:
    print(f'  {source:7} test {test_id}  train {train_id}  ratio={sim:.3f}')

within_test = []
for i, a in enumerate(test):
    for b in test[i+1:]:
        if a['source'] != b['source']:
            continue
        sim = SequenceMatcher(None, a['text'], b['text'], autojunk=False).ratio()
        if sim >= 0.92:
            within_test.append((sim, a['id'], b['id']))
within_test.sort(reverse=True)
print(f'\nNear-duplicate pairs within selected 40 held-out sections (>= 0.92): {len(within_test)}')
for sim, id_a, id_b in within_test[:20]:
    print(f'  {id_a} / {id_b}  ratio={sim:.3f}')

print('\nNote: zero exact/near matches is reassuring, not proof of zero contamination.')
print('A match is a flag for manual review, not automatic proof of leakage.')
