"""SFT examples for the 1.5B rewriter from pair files, with optional control tokens.

Inputs:  pair files in the radio_ready.pairs format; default pairs/final.jsonl (lyric
         sections) and data/processed/paradehate/pairs.jsonl (radio_ready.paradehate)
Outputs: data/processed/sft/<name>.jsonl, one example per pair:
           id, source, split, controls, prompt, completion
         data/processed/sft/<name>.tokens.json: the closed-vocabulary control tokens used,
           for tokenizer.add_special_tokens (open-vocabulary ones such as end sounds are
           left as text)

The prompt is the source's instruction (INSTRUCTIONS) followed by the original lines, one per
row, each preceded by its control tokens. The completion is the rewritten lines, one per row
(unchanged lines copied), so the loss can be put on the completion only. Control values are
computed from the original lines, which are known at inference time.

Controls (--controls, any of):
  edit    <edit> on lines with a detected explicit word (radio_ready.explicit), i.e. the
          lines to rewrite
  syl     <syl:N>  syllables of the original line (radio_ready.text; capped at MAX_SYLLABLES)
  rhyme   <rhyme:X>  the line's letter in the original section's rhyme scheme
  end     <end:KEY>  the line's end-rhyme sound (CMU stressed vowel key, open vocabulary)
  stress  <stress:0101>  CMU stress pattern of the line (open vocabulary)
ParaDeHate pairs (stage 1) get no control tokens: their rewrites change length freely (12%
of rewritten lines stay within one syllable of the original) and 46% of rewritten lines have no
detected explicit word, so <syl:N> and <edit> there would teach the model to ignore the tokens.
Stage 1 teaches detoxification only; the tokens are learned on the lyric pairs (stage 2), where
they hold (92% within one syllable; <edit> marks the rewritten lines). Token spelling:
TOKEN_FORMAT.

Run:  python -m radio_ready.sft                                   # no control tokens
      python -m radio_ready.sft --controls edit syl rhyme
      python -m radio_ready.sft pairs/final.jsonl --controls syl rhyme --name lyrics_syl_rhyme
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from radio_ready import paths
from radio_ready.explicit import ExplicitDetector
from radio_ready.text import line_rhyme_keys, rhyme_scheme, stress, syllables

TOKEN_FORMAT = "<{name}:{value}>"
EDIT_TOKEN = "<edit>"
MAX_SYLLABLES = 30
CLOSED = {"edit", "syl", "rhyme"}  # finite token sets, added to the tokenizer
LYRIC_ONLY = {"rhyme", "end", "stress"}  # skipped for single-line pairs
CONTROLS = ["edit", "syl", "rhyme", "end", "stress"]

INSTRUCTIONS = {
    "lyrics": "Rewrite these lyrics as a clean radio edit. Keep the line count, meaning, "
              "syllables, rhyme, slang and dialect.",
    "paradehate": "Rewrite this text without hateful or explicit language. Keep the meaning.",
}

DEFAULT_INPUTS = [paths.ROOT / "pairs" / "final.jsonl", paths.PARADEHATE_PAIRS]


def control_tokens(lines: list[str], controls: list[str], detector) -> list[list[str]]:
    """Control tokens per line, in the order of --controls."""
    single = len(lines) == 1
    scheme = rhyme_scheme(lines) if "rhyme" in controls and not single else ""
    out = []
    for i, line in enumerate(lines):
        tokens = []
        for name in controls:
            if single and name in LYRIC_ONLY:
                continue
            if name == "edit":
                if detector.find(line):
                    tokens.append(EDIT_TOKEN)
            elif name == "syl":
                tokens.append(TOKEN_FORMAT.format(name="syl",
                                                  value=min(syllables(line), MAX_SYLLABLES)))
            elif name == "rhyme":
                tokens.append(TOKEN_FORMAT.format(name="rhyme", value=scheme[i]))
            elif name == "end":
                keys = line_rhyme_keys(line)
                tokens.append(TOKEN_FORMAT.format(name="end", value=keys[0] if keys else "-"))
            elif name == "stress":
                tokens.append(TOKEN_FORMAT.format(name="stress", value=stress(line) or "-"))
        out.append(tokens)
    return out


def example(pair: dict, controls: list[str], detector) -> dict:
    kind = "paradehate" if pair["source"] == "paradehate" else "lyrics"
    if kind == "paradehate":
        controls = []  # see the module docstring
    tokens = control_tokens(pair["original"], controls, detector)
    rows = [" ".join(t + [line]) for t, line in zip(tokens, pair["original"])]
    return {
        "id": pair["id"], "source": pair["source"], "split": pair["split"],
        "controls": controls,
        "prompt": INSTRUCTIONS[kind] + "\n" + "\n".join(rows) + "\n",
        "completion": "\n".join(pair["rewrite"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="SFT examples with optional control tokens.")
    parser.add_argument("inputs", nargs="*", default=[str(p) for p in DEFAULT_INPUTS])
    parser.add_argument("--controls", nargs="*", choices=CONTROLS, default=[])
    parser.add_argument("--name", default=None)
    args = parser.parse_args()

    name = args.name or "sft_" + ("_".join(args.controls) if args.controls else "plain")
    detector = ExplicitDetector()
    examples, closed = [], set()
    for path in map(Path, args.inputs):
        for pair in map(json.loads, open(path, encoding="utf-8")):
            ex = example(pair, args.controls, detector)
            examples.append(ex)
            closed |= {t for row in ex["prompt"].splitlines() for t in row.split()
                       if t == EDIT_TOKEN or t.startswith(("<syl:", "<rhyme:"))}

    paths.SFT_DIR.mkdir(parents=True, exist_ok=True)
    out = paths.SFT_DIR / f"{name}.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    tokens_file = paths.SFT_DIR / f"{name}.tokens.json"
    tokens_file.write_text(json.dumps(sorted(closed), indent=1), encoding="utf-8")

    counts = Counter((e["source"], e["split"]) for e in examples)
    print(f"examples: {len(examples)}; by source and split: "
          + ", ".join(f"{s}/{sp} {n}" for (s, sp), n in sorted(counts.items())))
    print(f"control tokens to add: {len(closed)}; outputs: {out}, {tokens_file}")


if __name__ == "__main__":
    main()
