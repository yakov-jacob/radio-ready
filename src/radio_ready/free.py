"""Freer radio edits with Qwen2.5-7B: the model reads the 4-line section and rewrites the
flagged lines as it needs to, keeping rhyme and meter.

One call per 4-line block that has a flagged line (a line with a detected explicit word).
The model sees the whole block, the flagged lines marked with their censored words, each
flagged line's syllable count and the words it rhymes with, plus a few edits you liked
(from annotations/vetting.csv, never from the song being rewritten). It may reword a flagged
line freely; it must drop the censored words, keep what the line says in the verse, its
syllables and its end rhyme, and keep the voice, slang and dialect. Lines without a flagged
word are always kept exactly as written (any change the model makes to them is undone).

Candidates: one greedy answer and --samples sampled ones (temperature 1.0, top-p 0.9), scored
per flagged line. A rewrite of a line is out if it still has a detected explicit word, is
toxic (>= 0.9), is unchanged, or brings in a word that is not in the original line, the CMU
dictionary or clean lyrics (seen 20+ times; slang such as "homies" is fine, "gud" is not).
The rest are ranked: within one syllable, end rhyme kept, then meaning (similarity to the
original line) x fluency (CoLA, relative to the original). Lines with no clean rewrite get
--retries new calls, the block shown with the rewrites already accepted. A line with no
clean rewrite after that gets no proposal: an explicit line is never proposed.

--version 2 also names the stock swaps of the vetting sheet (generic replacements
used 5+ times for an explicit word: shit -> stuff, bitch -> girl, fuck -> mess, ...) and asks
for a creative rewrite of the phrase instead; rewrites that bring in one of them rank below
those that do not, and the style examples are liked edits without one. --version 1 is the
first prompt.

--version 3 (default) mines the stock swaps automatically instead (stock_table): every replacement of an
explicit word in every generation run, the vetting sheet, the Kidz Bop substitutions and the
official clean versions, counted per explicit word. The prompt names the stock swaps of the
words flagged in the block; a rewrite that brings in a stock swap of its line's explicit word,
or one generic everywhere, ranks below those that do not.

Output: outputs/generations/free_<input>_<VERSION>.jsonl in the radio_ready.generate format;
the vetting app lists it as its own source.

Run:  python -m radio_ready.free [--input FILE] [--offset 0] [--limit 10] [--version 2]
                                [--samples 4] [--retries 1]
"""

import argparse
import difflib
import json
import math
import random
import re
import time
from collections import Counter
from functools import lru_cache
from pathlib import Path

from radio_ready import cmu, paths
from radio_ready.commercial import reserved_ids
from radio_ready.explicit import ExplicitDetector
from radio_ready.prompts import clean_response_line, target_lines
from radio_ready.rewrite import load, rhyme_instruction, unload
from radio_ready.scoring import TOXICITY_THRESHOLD, Fluency, Similarity, Toxicity
from radio_ready.text import end_rhyme_kept, syllables

VERSIONS = {1: "free-v1", 2: "free-v2", 3: "free-v3"}
# free-v4 (one free rewrite of each line) was dropped as a method; its runs stay as data
MODEL = "Qwen/Qwen2.5-7B-Instruct"
N_EXAMPLES = 6

SYSTEM = ("You are a professional lyricist who writes radio edits of rap and pop songs. You "
          "rewrite the lines radio would censor so they sound like the artist wrote them clean, "
          "and you leave every other line exactly as it is.")

PROMPT = """\
Four lines from "{title}" by {artist}:

{block}

The lines marked >>> have words radio would censor:
{flagged}

Rewrite the marked lines for the radio edit. Read all four lines first to understand what the \
verse says. You can reword a marked line as much as you need, but:
- every censored word must be gone, with no other censored word, slur or sound-alike in its \
place;
- keep what the line means in this verse, in the artist's voice, slang and dialect (do not \
fix grammar or spelling);
- keep its syllable count and its end rhyme as given above, so it flows like the original;
- use real words only.
{creative}Copy the unmarked lines exactly as they are.

Radio edits of other songs, for the style:
{examples}

Answer with the four lines, numbered 1. to 4., and nothing else."""

CREATIVE = """Do not just swap a censored word for a stock substitute ({generic}): those make every radio edit sound the same. Rework the phrase around it with fresh words or an image that fits this verse, the way the artist could have written it clean.
"""

STOP = {"it", "the", "a", "an", "you", "so", "your", "to", "with", "i", "me", "that", "no",
        "be", "up", "get", "got", "yeah", "my", "of", "in", "on", "and", "is", "this"}
MIN_SWAPS = 5

NUMBERED = re.compile(r"^\s*(?:>>>\s*)?([1-4])[.):]\s*(.*)$")
WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


@lru_cache(maxsize=1)
def sheet_rows() -> list[dict]:
    from radio_ready.pairs import load_sections, load_sheet

    return [r for r in load_sheet(load_sections()) if r.get("original")]


def swaps(original: str, new: str, detector) -> list[tuple[str, str]]:
    """(explicit words, replacement) for each replaced span of `new` that held one."""
    a, b = original.lower().split(), new.lower().split()
    out = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        src = " ".join(a[i1:i2])
        if op == "replace" and detector.find(src):
            out.append((src, " ".join(w.strip(",.!?'\"") for w in b[j1:j2])))
    return out


@lru_cache(maxsize=1)
def generic_words() -> tuple[str, ...]:
    """Replacement words used MIN_SWAPS+ times for an explicit word in the vetting sheet (every
    proposal and edit): the stock swaps, most frequent first."""
    detector, counts = ExplicitDetector(), Counter()
    for r in sheet_rows():
        new = (r.get("edit") or "").strip() or r.get("rewrite", "")
        for _, dst in swaps(r["original"], new, detector) if new else []:
            if len(dst.split()) == 1 and dst not in STOP and dst and not detector.find(dst):
                counts[dst] += 1
    return tuple(w for w, n in counts.most_common() if n >= MIN_SWAPS)


def generic_in(new: str, original: str) -> list[str]:
    """Stock-swap words the rewrite brings in (not already in the original line)."""
    have = {w.lower() for w in WORD.findall(original)}
    return [w for w in WORD.findall(new) if w.lower() not in have
            and w.lower() in set(generic_words())]


# ---- v3: stock swaps mined automatically, per explicit word ---------------------------------
# Every replacement of an explicit word is counted, from every candidate of every generation
# run (vetted or not), every proposal and edit in the vetting sheet, the Kidz Bop substitutions
# and the official clean versions. A single-word replacement used for STOCK_MIN_LINES or more
# different lines of one explicit word (inflections merged: fuckin, fucked -> fuck) is a stock
# swap of that word; one that is a stock swap of GLOBAL_MIN_WORDS or more explicit words is
# generic for every line.

STOCK_MIN_LINES = 3
GLOBAL_MIN_WORDS = 3
STOCK_IN_PROMPT = 8


def explicit_key(word: str) -> str:
    w = word.lower().strip("'")
    base = re.sub(r"(?:in'|ing|in|ed|ers|er|es|s)$", "", w)
    return base if len(base) >= 3 else w


def _observations(detector):
    """(line key, explicit word, replacement) for every observed replacement."""
    from radio_ready.pairs import load_sections

    sections = load_sections()
    for r in sheet_rows():
        new = (r.get("edit") or "").strip() or r.get("rewrite", "")
        if new:
            yield (r["id"], r.get("line")), r["original"], new
    for path in sorted(paths.GENERATIONS_DIR.glob("*.jsonl")):
        for rec in map(json.loads, open(path, encoding="utf-8")):
            found = sections.get(rec.get("id"))
            if not found:
                continue
            orig = found[1]["lines"]
            for u in rec.get("units") or []:
                start, end = u.get("start", 0), u.get("end", 0)
                for c in u.get("candidates") or []:
                    if c.get("line"):
                        yield (rec["id"], start), orig[start], c["line"]
                    for k, text in enumerate(c.get("lines") or []):
                        if start + k < min(end, len(orig)) and text:
                            yield (rec["id"], start + k), orig[start + k], text
    kb = paths.KIDZBOP_SUBSTITUTIONS_CSV
    if kb.exists():
        import csv
        for row in csv.DictReader(open(kb, encoding="utf-8")):
            for n in range(int(row["count"])):
                yield ("kidzbop", row["badword"], n), row["badword"], row["replacement"]
    clean = paths.PROCESSED / "genius_clean" / "lines.jsonl"
    if clean.exists():
        for rec in map(json.loads, open(clean, encoding="utf-8")):
            if rec.get("kind") == "rewritten":
                yield ("clean", rec["id"]), rec["original"], rec["clean"]


@lru_cache(maxsize=1)
def stock_table() -> tuple[dict[str, list[str]], frozenset[str]]:
    """{explicit word key: stock swaps, most used first}, and the words generic everywhere."""
    detector = ExplicitDetector()
    lines_by: dict[tuple[str, str], set] = {}
    for key, original, new in _observations(detector):
        for src, dst in swaps(original, new, detector):
            dst = dst.strip()
            if (len(dst.split()) != 1 or not dst or dst in STOP or detector.find(dst)
                    or not WORD.fullmatch(dst)):
                continue
            for e in detector.find(src):
                lines_by.setdefault((explicit_key(e), dst), set()).add((key, dst))
    per_word: dict[str, list[tuple[int, str]]] = {}
    for (e, dst), seen in lines_by.items():
        if len(seen) >= STOCK_MIN_LINES:
            per_word.setdefault(e, []).append((len(seen), dst))
    stock = {e: [w for _, w in sorted(ws, reverse=True)] for e, ws in per_word.items()}
    spread = Counter(w for ws in stock.values() for w in ws)
    everywhere = frozenset({w for w, n in spread.items() if n >= GLOBAL_MIN_WORDS}
                           | set(generic_words()))
    return stock, everywhere


def stock_for(original: str, detector) -> set[str]:
    stock, everywhere = stock_table()
    out = set(everywhere)
    for e in detector.find(original):
        out |= set(stock.get(explicit_key(e), []))
    return out


def generic_v3(new: str, original: str, detector) -> list[str]:
    """Stock swaps (for this line's explicit words, or generic everywhere) the rewrite brings in."""
    have = {w.lower() for w in WORD.findall(original)}
    stock = stock_for(original, detector)
    return [w for w in WORD.findall(new) if w.lower() not in have and w.lower() in stock]


CREATIVE_V3 = """Do not just swap a censored word for a stock substitute: those make every radio edit sound the same. Stock substitutes to avoid here: {generic}. Rework the phrase around the censored word with fresh words or an image that fits this verse, the way the artist could have written it clean.
"""


def added(new: str, original: str) -> set[str]:
    """Content words a rewrite brings in that its original line does not have."""
    have = {w.lower() for w in WORD.findall(original)}
    return {w.lower() for w in WORD.findall(new)} - have - STOP


def changed_share(new: str, original: str) -> float:
    """Share of the original line's words the rewrite changes (word alignment)."""
    o, n = [w.lower() for w in WORD.findall(original)], [w.lower() for w in WORD.findall(new)]
    if not o:
        return 0.0
    kept = sum(b.size for b in difflib.SequenceMatcher(None, o, n, autojunk=False)
               .get_matching_blocks())
    return round(1 - kept / len(o), 3)


ANSWER_PREFIX = re.compile(r"^\s*(?:>>>\s*)?(?:\d+[.):]\s*)?")
# standard spellings that replace an artist's spelling (gon -> gonna, wit -> with, fuckin' ->
# messing): counted on every candidate, a soft tie-break in rank (it never fails a line)
STANDARD = {"gon": {"gonna", "going"}, "wit": {"with"}, "dat": {"that"}, "dem": {"them"},
            "'em": {"them"}, "em": {"them"}, "ya": {"you", "your"}, "tryna": {"trying"},
            "finna": {"fixing"}, "ain't": {"isn't", "aren't", "am"}, "y'all": {"you"},
            "lil": {"little"}, "cuz": {"because"}, "'cause": {"because"}, "da": {"the"}}


def answer_line(text: str) -> str | None:
    """The first line of an answer; quotes are removed only when they wrap the whole line."""
    text = text.replace("�", "'").replace("’", "'").replace("‘", "'")
    text = text.replace("“", '"').replace("”", '"')  # curly or broken quotes
    for row in text.strip().splitlines():
        row = clean_response_line(ANSWER_PREFIX.sub("", row.strip().strip("`")))
        if len(row) > 1 and row[0] == row[-1] and row[0] in "\"'":
            row = row[1:-1].strip()
        if row:
            return row
    return None


def spelling_lost(new: str, original: str) -> list[str]:
    """Standard spellings the rewrite uses where the original has the artist's own."""
    o = [w.lower() for w in re.findall(r"[A-Za-z']+", original)]
    n = {w.lower() for w in re.findall(r"[A-Za-z']+", new)}
    lost = {s for w in o for s in STANDARD.get(w, ())} & n
    if any(w.endswith("in'") for w in o):  # dropped g: -ing words the original did not have
        lost |= {w for w in n if w.endswith("ing") and len(w) > 4} - set(o)
    return sorted(lost)


@lru_cache(maxsize=1)
def substitution_sections() -> tuple[dict[str, set[str]], int]:
    """{word: ids of the sections whose proposals in the vetting sheet add it}, number of
    sections: stock swaps are added in many sections, by many artists."""
    found: dict[str, set[str]] = {}
    ids = set()
    for r in sheet_rows():
        if r.get("rewrite"):
            ids.add(r["id"])
            for w in added(r["rewrite"], r.get("original", "")):
                found.setdefault(w, set()).add(r["id"])
    return found, max(1, len(ids))


def originality(new: str, original: str, sid: str) -> float:
    """Mean surprisal of the words the rewrite adds, under how many other sections' rewrites
    (other songs and artists) add them; the section's own rewrites do not count."""
    words = added(new, original)
    if not words:
        return 0.0
    found, total = substitution_sections()
    return round(sum(-math.log((len(found.get(w, set()) - {sid}) + 1) / total) for w in words)
                 / len(words), 3)


def vocabulary(new: str, original: str) -> float:
    """Share of the added words that are lyric vocabulary (seen in clean lyrics)."""
    words = added(new, original)
    if not words:
        return 1.0
    known = lyric_words()
    return round(sum(w in known for w in words) / len(words), 3)


def creative_v3(censored: list[str]) -> str:
    stock, _ = stock_table()
    parts = []
    for e in sorted({explicit_key(w) for w in censored}):
        ws = stock.get(e, [])[:STOCK_IN_PROMPT]
        if ws:
            parts.append(f'"{e}" -> ' + ", ".join(ws))
    return CREATIVE_V3.format(generic="; ".join(parts) or ", ".join(
        f'"{w}"' for w in generic_words()[:12]))


def examples(exclude_ids: set[str], k: int = N_EXAMPLES,
             creative: bool = False) -> list[tuple[str, str]]:
    """Up to k (original, edit) line pairs you liked, your own edits first, one per song;
    `creative`: none that brings in a stock-swap word."""
    rows = [r for r in sheet_rows()
            if r.get("verdict") in ("like", "superlike") and r["id"] not in exclude_ids]
    edited = [r for r in rows if (r.get("edit") or "").strip()]
    plain = [r for r in rows if not (r.get("edit") or "").strip() and r.get("rewrite")
             and r.get("verdict") == "superlike"]
    out, seen, detector = [], set(), ExplicitDetector()
    for r in edited + plain:
        new = (r.get("edit") or "").strip() or r["rewrite"]
        if r["id"] in seen or new == r["original"]:
            continue
        if creative and (generic_in(new, r["original"])
                         or generic_v3(new, r["original"], detector)):
            continue
        seen.add(r["id"])
        out.append((r["original"], new))
        if len(out) == k:
            break
    return out


def block_text(lines: list[str], start: int, targets: dict[int, list[str]], record,
               detector) -> tuple[str, str]:
    rows, flagged = [], []
    for j in range(start, min(len(lines), start + 4)):
        n = j - start + 1
        rows.append(f"{'>>> ' if j in targets else '    '}{n}. {lines[j]}")
        if j in targets:
            words = ", ".join(f'"{w}"' for w in targets[j])
            flagged.append(f"- line {n}: {words}; {syllables(record['lines'][j])} syllables; "
                           f"{rhyme_instruction(record, j, detector).rstrip(';')}")
    return "\n".join(rows), "\n".join(flagged)


def parse(text: str, block: list[str]) -> list[str] | None:
    """The four numbered lines of an answer; None if any is missing."""
    got = {}
    for row in text.strip().splitlines():
        m = NUMBERED.match(row)
        if m and int(m.group(1)) not in got:
            got[int(m.group(1))] = clean_response_line(m.group(2).strip().strip("`\"'")).strip()
    if any(not got.get(n) for n in range(1, len(block) + 1)):
        return None
    return [got[n] for n in range(1, len(block) + 1)]


@lru_cache(maxsize=1)
def lyric_words(min_count: int = 20) -> frozenset[str]:
    """Words seen at least min_count times in clean lyrics: slang counts as real words
    (homies, hunnids, winnin), typos and made-up words do not."""
    from radio_ready.rhymes import lyric_counts

    return frozenset(w for w, n in lyric_counts().items() if n >= min_count)


def non_words(new: str, original: str) -> list[str]:
    """Words of the rewrite that are not in the original, the CMU dictionary or clean lyrics."""
    have = {w.lower() for w in WORD.findall(original)}
    return [w for w in WORD.findall(new) if w.lower() not in have
            and not cmu.phones_for_word(w.lower()) and w.lower() not in lyric_words()]


def stem(word: str) -> str:
    return re.sub(r"(?:es|s|e)$", "", word.lower()) or word.lower()


def phones(word: str) -> tuple[str, ...]:
    p = cmu.phones_for_word(word.lower())
    return tuple(re.sub(r"\d", "", x) for x in p[0].split()) if p else ()


def disguised(new: str, original: str, censored: list[str]) -> list[str]:
    """New words that are the censored word in another spelling or a homophone of it, which
    the detector can miss (hoe -> ho, hos)."""
    have = {w.lower() for w in WORD.findall(original)}
    out = []
    for w in WORD.findall(new):
        if w.lower() in have:
            continue
        for e in censored:
            same_sound = phones(w) and phones(w) in (phones(e), phones(stem(e)))
            if stem(w) == stem(e) or same_sound:
                out.append(w)
                break
    return out


class Scorer:
    def __init__(self, detector):
        self.detector = detector
        self.similarity = Similarity(on="cpu")
        self.toxicity = Toxicity(on="cpu")
        self.fluency = Fluency(on="cpu")

    def score(self, new: list[str], old: list[str]) -> list[dict]:
        """One score dict per (rewrite, original line)."""
        sims = self.similarity.pairwise(new, old)
        toxs = self.toxicity.score(new)
        flu = self.fluency.score(old + new)
        out = []
        for n in range(len(new)):
            censored = self.detector.find(old[n])
            c = {"line": new[n], "explicit": sorted(set(self.detector.find(new[n]))
                                                    | set(disguised(new[n], old[n], censored))),
                 "toxicity": round(float(toxs[n]), 3), "similarity": round(float(sims[n]), 3),
                 "syllable_diff": abs(syllables(new[n]) - syllables(old[n])),
                 "rhymes": end_rhyme_kept(old[n], new[n]),
                 "fluency_drop": round(max(float(flu[n] - flu[len(old) + n]), 0.0), 3),
                 "non_words": non_words(new[n], old[n]), "unchanged": new[n] == old[n],
                 "generic": generic_in(new[n], old[n]),
                 "generic_v3": generic_v3(new[n], old[n], self.detector)}
            c["ok"] = bool(not c["explicit"] and c["toxicity"] < TOXICITY_THRESHOLD
                           and not c["non_words"] and not c["unchanged"])
            out.append(c)
        return out


def rank(c: dict, version: int = 2) -> tuple:
    if version >= 3:
        generic = bool(c.get("generic_v3"))
    else:
        generic = bool(c.get("generic")) if version >= 2 else False
    return (not c["ok"], generic, c["syllable_diff"] > 1, not c["rhymes"],
            -c["similarity"] * (1 - c["fluency_drop"]))


def ask(tok, model, scorer, record, lines, start, todo, shot_text, samples, attempt,
        version=2):
    """Scored rewrites {line index: [candidates]} of the lines `todo` of one 4-line block,
    shown with the current `lines` (rewrites already accepted are filled in)."""
    from radio_ready.generate import complete

    detector = scorer.detector
    targets = {i: sorted(set(detector.find(record["lines"][i]))) for i in todo}
    shown, notes = block_text(lines, start, targets, record, detector)
    creative = (creative_v3([w for i in todo for w in targets[i]]) if version >= 3
                else CREATIVE.format(generic=", ".join(f'"{w}"' for w in generic_words()[:12]))
                if version >= 2 else "")
    user = PROMPT.format(title=record["title"], artist=record["artist"], block=shown,
                         flagged=notes, examples=shot_text, creative=creative)
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    raws = complete(tok, model, msgs, 1, 256, 0.0, 1.0, False)
    if samples:
        raws += complete(tok, model, msgs, samples, 256, 1.0, 0.9, False)
    block = lines[start:start + 4]
    found: dict[int, list[dict]] = {i: [] for i in todo}
    for k, raw in enumerate(raws):
        got = parse(raw, block)
        if not got:
            continue
        scored = scorer.score([got[i - start] for i in todo], [record["lines"][i] for i in todo])
        for i, c in zip(todo, scored):
            c.update(sample="greedy" if k == 0 else f"t1.0#{k}", attempt=attempt)
            found[i].append(c)
    return found, raws


DENSE_WORDS = 3  # a line saturated with profanity: this many explicit words, or
DENSE_SHARE = 0.25  # at least 2 that make up this share of its words 


def dense(line: str, detector) -> bool:
    n, words = len(detector.find(line)), len(WORD.findall(line))
    return n >= DENSE_WORDS or (n >= 2 and words > 0 and n / words >= DENSE_SHARE)


def open_lines(detector, include_pending: bool = False
               ) -> tuple[dict[str, dict], dict[str, list[int]]]:
    """Sections of the vetting sheet and, per section, its explicit lines still open: no
    accepted version in any source, and no proposal waiting for a verdict other than garbage
    (no change or nonsense). That is every line whose proposals were all rejected, garbage, or
    asked in the human list. One line per group of repeated lines."""
    from radio_ready.pairs import (accepted_texts, garbage, load_sections, load_sheet,
                                   repeat_groups)

    sections = load_sections()
    rows = load_sheet(sections)
    accepted = accepted_texts(rows)
    pending = set() if include_pending else {
        (r["id"], int(r["line"])) for r in rows
        if r.get("rewrite") and not r.get("verdict", "").strip() and not garbage(r)}
    records, found = {}, {}
    for sid in sorted({r["id"] for r in rows}):
        if sid not in sections:
            continue
        lines = sections[sid][1]["lines"]
        explicit = [i for i, l in enumerate(lines) if detector.find(l)]
        todo = [g[0] for g in repeat_groups(lines, explicit)
                if not any(accepted.get((sid, i + 1)) or (sid, i + 1) in pending for i in g)]
        if todo:  # shown with the accepted versions of its other lines (prompt context only)
            shown = [accepted[(sid, i + 1)][0][0] if accepted.get((sid, i + 1)) else l
                     for i, l in enumerate(lines)]
            records[sid], found[sid] = {**sections[sid][1], "shown": shown}, todo
    return records, found


def generated_ids() -> set[str]:
    """Ids of every section any generation run (any method, any model) already covers."""
    done = set()
    for f in paths.GENERATIONS_DIR.glob("*.jsonl"):
        for line in open(f, encoding="utf-8"):
            m = re.search(r'"id": "([0-9a-f]+)"', line[:80])
            if m:
                done.add(m.group(1))
    return done


def main(method: dict | None = None) -> None:
    """Free runs, or with `method` a one-line-at-a-time method that shares the section
    selection, scoring and output: {prefix, version, description, ask, rank}."""
    parser = argparse.ArgumentParser(description=method["description"] if method
                                     else "Freer 7B radio edits of flagged lines.")
    parser.add_argument("--input", default=str(paths.GENIUS_SECTIONS_JSONL))
    parser.add_argument("--offset", type=int, default=0, help="skip the first N sections")
    parser.add_argument("--order", choices=["random", "file"], default="random",
                        help="random: a seeded shuffle of the sections (the file is sorted by "
                             "artist name, so taking it in order over-samples artists whose "
                             "names start with symbols or digits); file: the file's order")
    parser.add_argument("--seed", type=int, default=None,
                        help="shuffle seed (default: the current time, a new one per run; printed and saved "
                             "in every output record, so the run can be reproduced)")
    parser.add_argument("--symbol-names", action="store_true",
                        help="also sample artists whose names start with a symbol or digit "
                             "($lim, 03osc): 2.4%% of the sections but 63%% of the Genius "
                             "sections generated in file order, so left out by default")
    parser.add_argument("--limit", type=int, default=None,
                        help="sections (default 10; all with --missing)")
    if method is None:
        parser.add_argument("--version", type=int, choices=sorted(VERSIONS), default=3)
    parser.add_argument("--name", default=None)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--samples", type=int, default=None,
                        help="sampled rewrites per call (default 4)")
    parser.add_argument("--missing", action="store_true",
                        help="the open lines of the vetting sheet (rejected, garbage, human list) "
                             "instead of --input; --offset/--limit apply to their sections")
    parser.add_argument("--dense", action="store_true",
                        help="only lines saturated with profanity (>= 3 explicit words, or >= 2 "
                             "that are >= 25%% of the line)")
    parser.add_argument("--include-pending", action="store_true",
                        help="with --missing: also lines whose proposals still wait for a verdict")
    parser.add_argument("--retries", type=int, default=1,
                        help="new calls for lines with no clean rewrite yet")
    if method and method.get("add_args"):
        method["add_args"](parser)
    args = parser.parse_args()
    if method and method.get("setup"):
        method["setup"](args)
    if args.samples is None:
        args.samples = 4
    prefix = method["prefix"] if method else "free"
    label = method["version"] if method else VERSIONS[args.version]
    version = 0 if method else args.version

    detector = ExplicitDetector()
    skip = reserved_ids()
    if args.missing:
        found, open_targets = open_lines(detector, args.include_pending)
        records = [r for sid, r in found.items() if sid not in skip]
        if args.name is None and not args.dense:
            args.name = f"{prefix}_missing_{label}"
    else:
        open_targets = None
        records = [json.loads(l) for l in open(args.input, encoding="utf-8")]
        records = [r for r in records if r["id"] not in skip and target_lines(r, detector)]
        if not args.symbol_names:
            records = [r for r in records if r["artist"].strip()[:1].isalpha()]
        if args.order == "random":
            if args.seed is None:
                args.seed = int(time.time())  # a new seed every run, from the clock
            print(f"shuffle seed: {args.seed}", flush=True)
            random.Random(args.seed).shuffle(records)
        # never again on sections a run already covers (any method): every run adds new data
        done = generated_ids()
        records = [r for r in records if r["id"] not in done]
    if args.dense:
        if open_targets is None:
            open_targets = {r["id"]: target_lines(r, detector) for r in records}
        by_id = {r["id"]: r["lines"] for r in records}
        open_targets = {sid: [i for i in t if dense(by_id[sid][i], detector)]
                        for sid, t in open_targets.items() if sid in by_id}
        records = [r for r in records if open_targets.get(r["id"])]
        if args.name is None:
            args.name = f"{prefix}_{'missing_' if args.missing else ''}dense_{label}"
    if args.limit is None:
        args.limit = len(records) if args.missing else 10
    records = records[args.offset: args.offset + args.limit]
    shots = [] if method else examples({r["id"] for r in records}, creative=version >= 2)
    by_rank = method["rank"] if method else (lambda c: rank(c, version))  # noqa: E731
    shot_text = "\n".join(f'- "{a}" -> "{b}"' for a, b in shots) or "- (none yet)"
    scorer = Scorer(detector)
    lyric_words()
    tok, model = load(args.model, False)
    started = time.time()

    # a run of its own: the default name has the time, so a run never overwrites another
    name = args.name or f"{prefix}_{Path(args.input).stem}-{time.strftime('%Y%m%d-%H%M')}_{label}"
    out = paths.GENERATIONS_DIR / f"{name}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for n, record in enumerate(records, start=1):
            targets = (open_targets[record["id"]] if open_targets is not None
                       else target_lines(record, detector))
            lines = list(record.get("shown") or record["lines"])
            cands: dict[int, list[dict]] = {i: [] for i in targets}
            raws: list[str] = []
            for i in (sorted(targets) if method else []):
                for attempt in range(args.retries + 1):
                    found, raw = method["ask"](tok, model, scorer, record, lines, i,
                                               args.samples, attempt)
                    raws += raw
                    cands[i] += found[i]
                    best = min(cands[i], key=by_rank, default=None)
                    if best and best["ok"]:
                        lines[i] = best["line"]
                        break
            for start in ([] if method else sorted({i // 4 * 4 for i in targets})):
                todo = [i for i in sorted(targets) if start <= i < start + 4]
                for attempt in range(args.retries + 1):
                    found, raw = ask(tok, model, scorer, record, lines, start, todo,
                                     shot_text, args.samples, attempt, version)
                    raws += raw
                    for i, cs in found.items():
                        cands[i] += cs
                        best = min(cands[i], key=by_rank, default=None)
                        if best and best["ok"]:
                            lines[i] = best["line"]
                    todo = [i for i in todo if not any(c["ok"] for c in cands[i])]
                    if not todo:
                        break
            lines = [l if i in targets else o for i, (l, o) in enumerate(zip(lines, record["lines"]))]
            units = []
            for i in sorted(targets):
                order = sorted(range(len(cands[i])), key=lambda k: by_rank(cands[i][k]))
                chosen = order[0] if order and cands[i][order[0]]["ok"] else None
                units.append({"start": i, "end": i + 1, "chosen": chosen,
                              "candidates": cands[i]})
            done = [u for u in units if u["chosen"] is not None]
            passed = len(done) == len(units) and all(
                u["candidates"][u["chosen"]]["syllable_diff"] <= 1
                and u["candidates"][u["chosen"]]["rhymes"] for u in done)
            f.write(json.dumps({
                "id": record["id"], "model": args.model, "prompt_version": label,
                "seed": args.seed,
                "style": "free", "samples": args.samples, "temperature": 1.0,
                "targets": sorted(targets), "raw": raws,
                "rewrites": {str(i): l for i, (l, o) in enumerate(zip(lines, record["lines"]))
                             if l != o},
                "lines": lines, "valid": bool(done), "passed": passed, "units": units},
                ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{n}/{len(records)}] {record['id']} {len(done)}/{len(units)} lines rewritten",
                  flush=True)
    unload(model, tok)
    print(f"{len(records)} sections in {(time.time() - started) / 60:.1f} min, output: {out}")


if __name__ == "__main__":
    main()
