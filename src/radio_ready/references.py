"""Human-approved reference lines for evaluation, from the vetting sheet's line pairs.

pairs/lines.jsonl (radio_ready.pairs --combine or --rebuild) has one record per line someone
liked or superliked: id, line (1-based), split, original, rewrite, proposed_by. The references
for a section are its liked lines when the section is held out from training: its split is
test, or it is a section reserved for commercial-LLM evaluation.

`proposed_by` names who wrote the line (free-v2, commercial-claude, human, ...), so a system is
never scored against a line it proposed itself (radio_ready.compare).
"""

import json
import re
from collections import Counter

from radio_ready import paths

LINES_JSONL = paths.ROOT / "pairs" / "lines.jsonl"


def references(extra_ids=(), path=LINES_JSONL) -> dict[str, dict[int, list[tuple[str, str, str]]]]:
    """{section id: {0-based line: [(original line, reference line, proposed_by), ...]}}."""
    found: dict[str, dict[int, list[tuple[str, str, str]]]] = {}
    if not path.exists():
        return found
    extra = set(extra_ids)
    for rec in map(json.loads, open(path, encoding="utf-8")):
        if rec.get("split") != "test" and rec["id"] not in extra:
            continue
        found.setdefault(rec["id"], {}).setdefault(rec["line"] - 1, []).append(
            (rec["original"].strip(), rec["rewrite"].strip(),
             "; ".join(rec.get("proposed_by") or [])))
    return found


def _ngrams(text: str, k: int) -> Counter:
    return Counter(text[i : i + k] for i in range(len(text) - k + 1))


def chrf(hypothesis: str, reference: str, n: int = 6, beta: float = 2.0) -> float:
    """Character n-gram F-score (chrF, Popovic 2015: character n-grams 1..6, beta 2, spaces
    removed, lowercased), 0-100."""
    hyp, ref = re.sub(r"\s+", "", hypothesis.lower()), re.sub(r"\s+", "", reference.lower())
    precision, recall = [], []
    for k in range(1, n + 1):
        h, r = _ngrams(hyp, k), _ngrams(ref, k)
        if not h or not r:
            continue
        overlap = sum((h & r).values())
        precision.append(overlap / sum(h.values()))
        recall.append(overlap / sum(r.values()))
    if not precision:
        return 0.0
    p, r = sum(precision) / len(precision), sum(recall) / len(recall)
    return 0.0 if p + r == 0 else 100 * (1 + beta**2) * p * r / (beta**2 * p + r)
