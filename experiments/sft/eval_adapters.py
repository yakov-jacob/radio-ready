"""Score trained adapters on the held-out test-split lyric sections (greedy decoding).

Metrics (as in findings/sft_milestone.md): output line count right; on sections with valid
output, no detected explicit word left; on changed lines, exact syllable count and end rhyme
kept (radio_ready.rewrite.rhymes with the section's rhyme partners); and unflagged lines copied.
Each adapter is prompted exactly as it was trained (its config.json: controls, chat format).

Run:  python experiments/sft/eval_adapters.py ADAPTER [ADAPTER ...] [--limit N]
      (adapters are folders under outputs/sft_models)
Output: outputs/evaluation/adapters/<adapter>.jsonl (predictions) and a table on stdout.
"""

import argparse
import json

import torch

from radio_ready import paths
from radio_ready.explicit import ExplicitDetector
from radio_ready.rewrite import rhyme_partners, rhymes
from radio_ready.sft import example
from radio_ready.text import syllables
from radio_ready.train import MODELS_DIR, format_prompt, load_model, parse_completion

OUT = paths.OUTPUTS / "evaluation" / "adapters"


def test_sections() -> list[dict]:
    pairs = [json.loads(l) for l in open(paths.ROOT / "pairs" / "final.jsonl", encoding="utf-8")]
    seen, out = set(), []
    for p in pairs:
        if p["split"] == "test" and p["id"] not in seen:
            seen.add(p["id"])
            out.append(p)
    return out


@torch.no_grad()
def run(adapter: str, sections: list[dict], detector) -> list[dict]:
    cfg = json.loads((MODELS_DIR / adapter / "config.json").read_text())
    tok, model, device = load_model(cfg, MODELS_DIR / adapter, train=False)
    out = []
    for p in sections:
        prompt = format_prompt(tok, cfg, example(p, cfg["controls"], detector)["prompt"])
        enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(device)
        gen = model.generate(**enc, max_new_tokens=400, do_sample=False,
                             pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
        text = tok.decode(gen[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        out.append({"id": p["id"], "lines": parse_completion(text, len(p["original"])),
                    "raw": text})
    del model
    torch.cuda.empty_cache()
    return out


def score(sections: list[dict], preds: list[dict], detector) -> dict:
    c = {"sections": len(sections), "valid": 0, "clean": 0, "changed": 0, "syl_exact": 0,
         "rhyme": 0, "unflagged": 0, "unflagged_kept": 0}
    for p, pred in zip(sections, preds):
        lines = pred["lines"]
        if lines is None:
            continue
        c["valid"] += 1
        c["clean"] += not detector.find("\n".join(lines))
        rec = {"lines": p["original"]}
        for i, (o, w) in enumerate(zip(p["original"], lines)):
            if not detector.find(o):
                c["unflagged"] += 1
                c["unflagged_kept"] += w == o
            if w != o:
                c["changed"] += 1
                c["syl_exact"] += syllables(w) == syllables(o)
                c["rhyme"] += bool(rhymes(w, o, rhyme_partners(rec, i)))
    return c


def main() -> None:
    parser = argparse.ArgumentParser(description="Score adapters on test-split lyric sections.")
    parser.add_argument("adapters", nargs="+")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    detector = ExplicitDetector()
    sections = test_sections()[: args.limit]
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for a in args.adapters:
        preds = run(a, sections, detector)
        with open(OUT / f"{a}.jsonl", "w", encoding="utf-8") as f:
            for x in preds:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
        rows.append((a, score(sections, preds, detector)))
    print("| Adapter | Line count right | Clean | Exact syllables (changed) | Rhyme kept (changed) "
          "| Unflagged lines copied |")
    print("| --- | --- | --- | --- | --- | --- |")
    for a, c in rows:
        v, ch, un = max(c["valid"], 1), max(c["changed"], 1), max(c["unflagged"], 1)
        print(f"| {a} | {c['valid']}/{c['sections']} | {c['clean'] / v:.0%} | "
              f"{c['syl_exact'] / ch:.0%} | {c['rhyme'] / ch:.0%} | {c['unflagged_kept'] / un:.0%} |")


if __name__ == "__main__":
    main()
