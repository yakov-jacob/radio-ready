"""Held-out evaluation set of Genius sections, never used for training.

Two groups:
  commercial  every Genius section reserved for commercial LLMs (rows of
              data/commercial/commercial_table.csv)
  test        sections by test-split artists (the artist hash of radio_ready.labels, as in
              radio_ready.pairs): every one with a human-approved reference line
              (radio_ready.references) first, then a seeded sample of the rest up to
              --test-sections

radio_ready.train never trains on these: it uses split train only and drops reserved sections.

Inputs:  data/processed/genius_sections.jsonl, data/commercial/commercial_table.csv,
         pairs/lines.jsonl (optional, for references)
Output:  data/processed/eval_sections.jsonl: the section records plus eval_group and
         has_reference

Run:  python -m radio_ready.eval_set [--test-sections 250]
"""

import argparse
import json
import random
from collections import Counter

from radio_ready import paths
from radio_ready.commercial import reserved_ids
from radio_ready.labels import artist_split
from radio_ready.references import references

EVAL_SECTIONS_JSONL = paths.PROCESSED / "eval_sections.jsonl"
SYSTEMS_DIR = paths.OUTPUTS / "systems"  # rewrites of the eval set by baselines and trained models


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the held-out evaluation set.")
    parser.add_argument("--test-sections", type=int, default=250,
                        help="test-split sections, those with references first (in addition to "
                             "the reserved ones)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records = [json.loads(l) for l in open(paths.GENIUS_SECTIONS_JSONL, encoding="utf-8")
               if l.strip()]
    reserved = reserved_ids()
    refs = set(references(extra_ids=reserved))
    commercial = [dict(r, eval_group="commercial") for r in records if r["id"] in reserved]
    test = [r for r in records if r["id"] not in reserved and artist_split(r["artist"]) == "test"]
    with_ref = [r for r in test if r["id"] in refs]
    rest = [r for r in test if r["id"] not in refs]
    rng = random.Random(args.seed)
    test = with_ref + rng.sample(rest, max(0, min(args.test_sections - len(with_ref), len(rest))))
    chosen = commercial + [dict(r, eval_group="test") for r in test]
    chosen = [dict(r, has_reference=r["id"] in refs) for r in chosen]
    chosen.sort(key=lambda r: (r["eval_group"], r["id"]))

    EVAL_SECTIONS_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with open(EVAL_SECTIONS_JSONL, "w", encoding="utf-8") as f:
        for r in chosen:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"eval sections: {len(chosen)} {dict(Counter(r['eval_group'] for r in chosen))}, "
          f"with a reference: {sum(r['has_reference'] for r in chosen)}, "
          f"reserved sections not found: {len(reserved) - len(commercial)}")
    print(f"output: {EVAL_SECTIONS_JSONL}")


if __name__ == "__main__":
    main()
