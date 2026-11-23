"""Surgical rewrites of explicit lines that are still missing: a meaning draft (match) and a
singable rewrite of it (sing), as two separate sources.

Targets (--scope): explicit lines with no accepted version in any source and no proposal
waiting for a verdict (one line per group of repeated lines); `all` (default): in every section
a generation run covers, including lines whose every candidate failed the filters; `human`: the
vetting app's human list only.

Two passes with one model (default Qwen2.5-7B-Instruct, 4-bit):
1. Meaning draft: the 4-line stanza with the line marked and its explicit words named; the
   model replaces only those words with clean words that mean the same (greedy).
2. Sing: the stanza with the draft in place, the original line's syllable count and the end
   words it rhymes with; --samples answers rewrite the draft so it sings like the verse. The
   model never sees explicit text: other explicit lines of the stanza are shown with their
   first accepted version (any source), or with their draft.
Lines that an earlier sing-v2 run already drafted reuse its draft and candidates (no model).

Every candidate is checked like radio_ready.free (no explicit or disguised word, toxicity < 0.9,
real words, changed), must keep the meaning (cosine >= 0.70 to the draft, >= 0.60 to the original
line), must not copy another line of the section, and is never a text you already disliked for
that line. Outputs, in the radio_ready.generate format:
  match_<name>_match-v4.jsonl  the draft, when it passes (source match-v4: the most surgical)
  sing_<name>_sing-v3.jsonl    the best sung rewrite: end rhyme kept, within one syllable, then
                               meaning to the draft x fluency (source sing-v3)
Liked lines complete their sections (radio_ready.pairs.export_human).
Earlier runs: sing_<name>_sing-v2.jsonl (one pick per line, the draft or a sung rewrite);
--reselect applies the checks to them again from the stored scores, without a model.

Run:  python -m radio_ready.sing [--scope all|human] [--limit N] [--samples 4]
      python -m radio_ready.sing --reselect
"""

import argparse
import json
import time

from radio_ready import paths
from radio_ready.explicit import ExplicitDetector
from radio_ready.free import Scorer
from radio_ready.pairs import (
    HUMAN, HUMAN_REQUEST, accepted_texts, load_sections, load_sheet, repeat_groups, sources,
)
from radio_ready.prompts import clean_response_line
from radio_ready.rewrite import load, rhyme_instruction, rhyme_partners, rhymes, unload
from radio_ready.scoring import SIMILARITY_THRESHOLD
from radio_ready.text import syllables

VERSION = "sing-v2"  # earlier runs: one pick per line
SING_VERSION = "sing-v3"
MATCH_VERSION = "match-v4"
# Meaning to the original line: lower than to the draft, since dropping the explicit word moves
# a single line's embedding a lot ("Damn this shit fire" -> "Damn this stuff lit": 0.68); it
# still rejects a replaced or copied line (0.3-0.5).
MIN_SIMILARITY_TO_ORIGINAL = 0.60
MODEL = "Qwen/Qwen2.5-7B-Instruct"

SYSTEM = ("You are a professional lyricist who writes radio edits of songs. You change only what "
          "radio would censor and keep everything else as the artist wrote it.")

DRAFT_PROMPT = """\
Lines from "{title}" by {artist}:

{stanza}

The line marked >>> has words that radio would censor: {words}.
Replace only those words with clean words that mean the same thing here, so the line and the \
stanza keep their meaning. Keep every other word exactly as it is, including slang and dialect.

Answer with the rewritten line only, without quotes or anything else."""

SING_PROMPT = """\
Lines from "{title}" by {artist}:

{stanza}

The line marked >>> was edited for radio and now sounds flat next to the others. Rewrite it so \
it sings like the rest of this verse:
- keep what it says, in the voice, slang and dialect of the verse;
- {syllables} syllables, with the same rhythm as the lines around it;
- {rhyme}
- no word radio would censor, and no sound-alike of one.

Answer with the rewritten line only, without quotes or anything else."""


def stanza(lines: list[str], i: int) -> str:
    start = (i // 4) * 4
    return "\n".join((">>> " if j == i else "    ") + lines[j]
                     for j in range(start, min(len(lines), start + 4)))


def first_line(text: str) -> str | None:
    rows = [clean_response_line(r.strip().strip("`\"'")) for r in text.strip().splitlines()]
    rows = [r.lstrip(">").strip().strip("\"'") for r in rows if r]
    return rows[0] if rows else None


def check(c: dict, i: int, section: list[str]) -> bool:
    """The candidate passes: free.py's checks, meaning kept (to the draft and to the original
    line) and not a copy of another line of the section."""
    others = {l.strip().lower() for j, l in enumerate(section) if j != i}
    return bool(c["ok"] and c["meaning_vs_draft"] >= SIMILARITY_THRESHOLD
                and c["similarity"] >= MIN_SIMILARITY_TO_ORIGINAL
                and c["line"].strip().lower() not in others)


def choose(scored: list[dict]) -> int | None:
    order = sorted(range(len(scored)), key=lambda k: (
        not scored[k]["passed"], not scored[k]["rhymes"], scored[k]["syllable_diff"] > 1,
        -scored[k]["meaning_vs_draft"] * (1 - scored[k]["fluency_drop"])))
    return order[0] if order and scored[order[0]]["passed"] else None


def reselect(path) -> None:
    sections = load_sections()
    gens = [json.loads(l) for l in open(path, encoding="utf-8")]
    before = sum(u["chosen"] is not None for g in gens for u in g["units"])
    for g in gens:
        original = sections[g["id"]][1]["lines"]
        final = list(original)
        for u in g["units"]:
            for c in u["candidates"]:
                c["passed"] = check(c, u["start"], original)
            u["chosen"] = choose(u["candidates"])
            if u["chosen"] is not None:
                final[u["start"]] = u["candidates"][u["chosen"]]["line"]
        g["lines"] = final
        g["rewrites"] = {str(i): l for i, (l, o) in enumerate(zip(final, original)) if l != o}
        done = [u for u in g["units"] if u["chosen"] is not None]
        g["valid"], g["passed"] = bool(done), len(done) == len(g["units"])
    with open(path, "w", encoding="utf-8") as f:
        for g in gens:
            f.write(json.dumps(g, ensure_ascii=False) + "\n")
    after = sum(u["chosen"] is not None for g in gens for u in g["units"])
    print(f"{path.name}: lines with a proposal {before} -> {after}")


def missing_lines(rows: list[dict], sections, scope: str, detector) -> dict[str, list[int]]:
    """{section id: line indexes} still open: no accepted version in any source and no proposal
    waiting for a verdict (one line per group of repeated lines).
      human  the lines of the vetting app's human list (their sections have accepted lines)
      all    every explicit line of every section a generation run covers"""
    accepted = accepted_texts(rows)
    pending = {(r["id"], int(r["line"])) for r in rows
               if r.get("rewrite") and not r.get("verdict", "").strip()}
    if scope == "human":
        found: dict[str, list[int]] = {}
        for r in rows:
            key = (r["id"], int(r["line"]))
            if (f"{HUMAN}: {HUMAN_REQUEST}" in r.get("sources", "") and not accepted.get(key)
                    and key not in pending):
                found.setdefault(r["id"], []).append(int(r["line"]) - 1)
        return {k: sorted(set(v)) for k, v in found.items()}
    covered = set()
    for spec in sources().values():
        for f in spec["files"]:
            covered |= {json.loads(l)["id"] for l in open(f, encoding="utf-8")}
    found = {}
    for sid in sorted(covered):
        if sid not in sections:
            continue
        lines = sections[sid][1]["lines"]
        explicit = [i for i, l in enumerate(lines) if detector.find(l)]
        for g in repeat_groups(lines, explicit):
            if any(accepted.get((sid, i + 1)) or (sid, i + 1) in pending for i in g):
                continue
            found.setdefault(sid, []).append(g[0])
    return found


def stored_units() -> dict[tuple[str, int], dict]:
    """Draft and candidates already generated for a line by earlier sing runs (reused)."""
    found = {}
    for path in sorted(paths.GENERATIONS_DIR.glob(f"sing_*_{VERSION}.jsonl")):
        for g in map(json.loads, open(path, encoding="utf-8")):
            for u in g["units"]:
                if u.get("candidates"):
                    found[(g["id"], u["start"])] = u
    return found


def record(sid, original, picks: dict[int, int | None], units: dict[int, dict], version, model,
           samples) -> dict:
    final = list(original)
    out_units = []
    for i, unit in sorted(units.items()):
        k = picks.get(i)
        if k is not None:
            final[i] = unit["candidates"][k]["line"]
        out_units.append({"start": i, "end": i + 1, "draft": unit.get("draft"), "chosen": k,
                          "candidates": unit["candidates"]})
    done = [u for u in out_units if u["chosen"] is not None]
    style = version.split("-")[0]
    return {"id": sid, "model": f"{style}+{model}", "prompt_version": version, "style": style,
            "samples": samples, "temperature": 1.0, "targets": sorted(units), "raw": "",
            "rewrites": {str(i): l for i, (l, o) in enumerate(zip(final, original)) if l != o},
            "lines": final, "valid": bool(done), "passed": len(done) == len(out_units),
            "units": out_units}


def main() -> None:
    parser = argparse.ArgumentParser(description="Surgical rewrites (match and sing) of missing "
                                                 "lines.")
    parser.add_argument("--scope", choices=["human", "all"], default="all")
    parser.add_argument("--limit", type=int, default=None, help="sections")
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--name", default=None)
    parser.add_argument("--reselect", action="store_true",
                        help="apply the checks and ranking again to existing sing-v2 runs")
    args = parser.parse_args()
    if args.reselect:
        for path in sorted(paths.GENERATIONS_DIR.glob(f"sing_*_{VERSION}.jsonl")):
            reselect(path)
        return

    detector = ExplicitDetector()
    sections = load_sections()
    rows = load_sheet(sections)
    todo = missing_lines(rows, sections, args.scope, detector)
    accepted = accepted_texts(rows)
    disliked = {(r["id"], int(r["line"]), (r.get("edit") or "").strip() or r.get("rewrite", ""))
                for r in rows if r.get("verdict", "").strip().lower() == "dislike"}
    stored = stored_units()
    ids = [i for i in sorted(todo) if i in sections][: args.limit]
    n_lines = sum(len(todo[i]) for i in ids)
    reuse = sum((i, j) in stored for i in ids for j in todo[i])
    print(f"sections: {len(ids)}, lines: {n_lines}, reused from earlier sing runs: {reuse}",
          flush=True)
    scorer = Scorer(detector)
    tok = model = None
    from radio_ready.generate import complete

    name = args.name or ("missing" if args.scope == "all" else "human-list")
    out_match = paths.GENERATIONS_DIR / f"match_{name}_{MATCH_VERSION}.jsonl"
    out_sing = paths.GENERATIONS_DIR / f"sing_{name}_{SING_VERSION}.jsonl"
    started = time.time()
    with open(out_match, "w", encoding="utf-8") as fm, open(out_sing, "w", encoding="utf-8") as fs:
        for n, sid in enumerate(ids, start=1):
            rec = sections[sid][1]
            original = rec["lines"]
            shown = list(original)  # the model never sees explicit text
            for j, line in enumerate(original):
                if detector.find(line) and accepted.get((sid, j + 1)):
                    shown[j] = accepted[(sid, j + 1)][0][0]
            units: dict[int, dict] = {i: stored[(sid, i)] for i in todo[sid] if (sid, i) in stored}
            new = [i for i in todo[sid] if i not in units]
            if new and model is None:
                tok, model = load(args.model, False)
            drafts = {}
            for i in new:
                words = sorted(set(detector.find(original[i])))
                msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content":
                         DRAFT_PROMPT.format(title=rec["title"], artist=rec["artist"],
                                             stanza=stanza(shown, i), words=", ".join(words))}]
                drafts[i] = first_line(complete(tok, model, msgs, 1, 128, 0.0, 1.0, False)[0])
            for i, u in units.items():
                shown[i] = u.get("draft") or shown[i]
            for i, d in drafts.items():
                if d:
                    shown[i] = d
            for i in new:
                draft = drafts.get(i)
                if not draft:
                    units[i] = {"draft": None, "candidates": []}
                    continue
                msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content":
                         SING_PROMPT.format(title=rec["title"], artist=rec["artist"],
                                            stanza=stanza(shown, i),
                                            syllables=syllables(original[i]),
                                            rhyme=rhyme_instruction(rec, i, detector))}]
                sung = [s for s in (first_line(r) for r in complete(
                    tok, model, msgs, args.samples, 128, 1.0, 0.9, False)) if s]
                cands = list(dict.fromkeys([draft] + sung))
                scored = scorer.score(cands, [original[i]] * len(cands))
                to_draft = scorer.similarity.pairwise(cands, [draft] * len(cands))
                partners = rhyme_partners(rec, i)
                for c, m in zip(scored, to_draft):
                    c["source"] = "draft" if c["line"] == draft else "sung"
                    c["meaning_vs_draft"] = round(float(m), 3)
                    c["rhymes"] = rhymes(c["line"], original[i], partners)
                units[i] = {"draft": draft, "candidates": scored}
            match_picks, sing_picks = {}, {}
            for i, u in units.items():
                for c in u["candidates"]:
                    c["passed"] = check(c, i, original) and (sid, i + 1, c["line"]) not in disliked
                drafts_ok = [k for k, c in enumerate(u["candidates"])
                             if c.get("source") == "draft" and c["passed"]]
                match_picks[i] = drafts_ok[0] if drafts_ok else None
                sung = [k for k, c in enumerate(u["candidates"]) if c.get("source") == "sung"]
                best = choose([u["candidates"][k] for k in sung]) if sung else None
                sing_picks[i] = sung[best] if best is not None else None
            fm.write(json.dumps(record(sid, original, match_picks, units, MATCH_VERSION,
                                       args.model, 0), ensure_ascii=False) + "\n")
            fs.write(json.dumps(record(sid, original, sing_picks, units, SING_VERSION,
                                       args.model, args.samples), ensure_ascii=False) + "\n")
            fm.flush()
            fs.flush()
            print(f"[{n}/{len(ids)}] {sid} match {sum(v is not None for v in match_picks.values())}"
                  f"/{len(units)}, sing {sum(v is not None for v in sing_picks.values())}"
                  f"/{len(units)}", flush=True)
    if model is not None:
        unload(model, tok)
    print(f"{len(ids)} sections in {(time.time() - started) / 60:.1f} min, outputs: {out_match}, "
          f"{out_sing}")


if __name__ == "__main__":
    main()
