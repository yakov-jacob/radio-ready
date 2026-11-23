"""Quick human vetting of synthetic rewrites, one line at a time (tkinter, no extra installs).

Double-click vet.bat: a start window lists every source (radio_ready.pairs.sources: every run
in outputs/generations, runs of one method grouped, whatever sections or batch they cover)
with how many of its lines you have vetted, plus "All sources". Enter or double-click opens
one. Nothing to export by hand: when the window opens and whenever you close a vetting window,
everything is updated in the background (radio_ready.pairs.update: new runs added, every
source's pairs, pairs/final and pairs/lines). A vetting window asked for meanwhile opens when
the update is done.

The vetting window shows one line: the original with its explicit words highlighted and the
proposed rewrite in an edit field. A rewrite that several sources proposed for the same line is
one row (annotations/vetting.csv), so it is vetted once. Change the text if you want: your
version is saved as a row of its own for the list you are vetting (radio_ready.pairs.
add_your_version) and becomes a pair of its own; the proposal keeps its verdict (a like on it
is never replaced), and another list shows the proposal, not your version. Then:

  Enter          like
  Ctrl+Enter     superlike
  Esc            dislike
  Up / Down      back to the previous line / skip this line (no verdict)
  Tab            context: the 4-line stanza, original and rewritten (closes on every line)
  Alt+1 ... 9    (Cmd+1 ... 9 on a Mac) put a suggestion in place of an explicit word: clean words that rhyme with
                 the explicit word itself, with its syllable count, never a sound-alike of it
                 (radio_ready.rhymes; scored by fit in the line and meaning, picked to differ
                 from each other; computed in the background). Every explicit word of the
                 line gets suggestions, numbered in order: 9 for one word, 4 each for two,
                 3 each for three, fewer as there are more

The "human" list holds the lines that keep a section from being paired: the section has accepted
lines, but this one has no accepted version in any source and no proposal waiting for your
verdict (it was disliked, or no model rewrote it). Write a clean version (Alt+1-9 for rhymes)
and press Enter or Ctrl+Enter; Esc skips. Finished sections are the pairs of the human source
(radio_ready.pairs.export_human).

The "garbage" list holds the rows no source list shows: proposals that change nothing (no
usable answer, or the original line back) and those every source marks nonsense. The sheet keeps
each source's status on them, so they still count as that model's failures (pairs --stats).

Repeated lines of a section are vetted once: a line is not asked when another occurrence of
it already has an accepted version, or when the same rewrite of it is already asked (pairs
use one version for every occurrence).

Every verdict is saved at once. All unvetted lines come first, the most original substitutions
before stock ones: ordered by how rarely the proposals in the sheet add the rewrite's new words
("girl", "stuff", "homie" come last), then by how rare they are in clean lyrics.
Then a pause screen: close the window, or press Enter to go on to the lines already vetted
(same order), to review or change them.

Run:  vet.bat (Windows) or vet.command (Mac)    # start window
      python -m radio_ready.vet [SOURCE]
"""

import argparse
import contextlib
import difflib
import io
import json
import math
import queue
import re
import sys
import threading
import tkinter as tk
from collections import Counter
from tkinter import font as tkfont

from radio_ready import paths
from radio_ready.explicit import ExplicitDetector
from radio_ready.pairs import (
    APPROVE, HUMAN, HUMAN_REQUEST, SECTION_FILES, YOURS, add_your_version, garbage, line_key,
    load_sheet, only_nonsense, source_names, sources, update, write_sheet,
)
from radio_ready.text import syllables

WORD = re.compile(r"[a-z']+")
CORPUS_FILES = ["corpus_200k.jsonl", "corpus.jsonl"]
ALL = "All sources"
WATCH_MS = 30_000  # how often the start window looks for new generation runs
GARBAGE = "garbage"  # one list for rows no source list shows: no change, or nonsense



def word_rarity() -> tuple[Counter, int]:
    for name in CORPUS_FILES:
        path = paths.PROCESSED / "match" / name
        if path.exists():
            counts: Counter = Counter()
            with open(path, encoding="utf-8") as f:
                for i, line in enumerate(f):
                    counts.update(WORD.findall(json.loads(line)["text"].lower()))
                    if i >= 300_000:
                        break
            return counts, sum(counts.values())
    return Counter(), 1


def added_words(original: str, rewrite: str) -> list[str]:
    old = set(WORD.findall(original.lower()))
    return [w for w in WORD.findall(rewrite.lower()) if w not in old]


def uniqueness(original: str, rewrite: str, counts: Counter, total: int) -> float:
    """Mean surprisal of the words the rewrite adds (rare words score high, "stuff" low)."""
    new = added_words(original, rewrite)
    if not new:
        return 0.0
    return sum(-math.log((counts[w] + 1) / total) for w in new) / len(new)


def substitution_counts(rows: list[dict]) -> tuple[Counter, int]:
    """How often each word is added by the proposals in the sheet (each distinct rewrite of a
    line once): the stock swaps every model reaches for ("stuff", "chick") count high."""
    counts, seen = Counter(), set()
    for r in rows:
        text = r.get("rewrite", "")
        key = (r["id"], r["line"], text)
        if text and key not in seen:
            seen.add(key)
            counts.update(set(added_words(r.get("original", ""), text)))
    return counts, max(1, len(seen))


def load_sections(ids: set[str] | None = None) -> dict[str, tuple[str, dict]]:
    """{id: (source, record)} for the given ids (all when None), read quickly."""
    found = {}
    for source, path in SECTION_FILES.items():
        if not path.exists():
            continue
        for line in open(path, encoding="utf-8"):
            m = re.search(r'"id": "([0-9a-f]+)"', line[:60])
            if m and (ids is None or m.group(1) in ids):
                found[m.group(1)] = (source, json.loads(line))
    return found


KEYS = 9  # Alt+1 ... Alt+9 (Cmd+1 ... Cmd+9 on a Mac, where Option+digit types a character)
MAC = sys.platform == "darwin"
SUGGEST_KEY = "Cmd" if MAC else "Alt"


def per_word(n_slots: int) -> int:
    """Suggestions per explicit word so that every word gets some within KEYS keys."""
    return max(1, KEYS // max(1, n_slots))


def slots(original: str, current: str, detector) -> list[dict]:
    """Where each explicit word of the original sits in the current line: the explicit word,
    the token span of current that replaces it, and trailing punctuation to keep."""
    a, b = original.split(" "), current.split(" ")
    ops = difflib.SequenceMatcher(None, [w.lower() for w in a], [w.lower() for w in b],
                                  autojunk=False).get_opcodes()
    found = []
    for i, token in enumerate(a):
        words = detector.find(token)
        if not words:
            continue
        for op, i1, i2, j1, j2 in ops:
            if i1 <= i < i2 or (op == "insert" and i1 == i):
                if op == "equal":
                    j1, j2 = j1 + (i - i1), j1 + (i - i1) + 1
                punct = re.search(r"[^\w']*$", b[j2 - 1] if j2 > j1 else token).group(0)
                found.append({"word": words[0], "start": j1, "end": j2, "punct": punct})
                break
    return found


class App:
    def __init__(self, root, rows, order, sections, detector, title, boundary=None):
        self.root, self.rows, self.order = root, rows, order
        # order[:boundary] are unvetted; a pause screen comes before the vetted ones
        self.boundary = len(order) if boundary is None else boundary
        self.paused, self.past_pause = False, False
        self.sections, self.detector, self.source = sections, detector, title
        self.pos, self.history, self.context = 0, [], False
        self.drafts: dict[int, str] = {}  # row index -> text typed but not voted on
        self.inserted: dict[int, set[str]] = {}  # row index -> suggestion words put in
        self.suggestions, self.results, self.pending, self.slots = [], queue.Queue(), None, []
        base = tkfont.nametofont("TkDefaultFont").actual()["family"]
        big = (base, 15)
        root.title(f"RadioReady vetting: {title}")
        root.geometry("1000x450")
        self.header = tk.Label(root, anchor="w", font=(base, 11))
        self.header.pack(fill="x", padx=12, pady=(10, 2))
        tk.Label(root, text="original", anchor="w", fg="#777").pack(fill="x", padx=12)
        self.orig = tk.Text(root, height=1, wrap="word", font=big, relief="flat", bg="#f4f4f4")
        self.orig.pack(fill="x", padx=12)
        self.orig.tag_configure("explicit", foreground="#c0392b", font=(base, 15, "bold"))
        tk.Label(root, text="rewrite (edit in place)", anchor="w", fg="#777").pack(
            fill="x", padx=12, pady=(8, 0))
        self.edit = tk.Text(root, height=1, wrap="word", font=big, undo=True)
        self.edit.pack(fill="x", padx=12)
        self.edit.bind("<KeyRelease>", lambda e: self.update_info())
        self.ctx = tk.Text(root, height=5, wrap="word", font=(base, 12), relief="flat",
                           bg="#f9f9f9")
        self.ctx.tag_configure("current", background="#fff3b0")
        self.info = tk.Label(root, anchor="w", fg="#555", font=(base, 11))
        self.info.pack(fill="x", padx=12, pady=(6, 0))
        self.rhymes = tk.Label(root, anchor="w", fg="#2c6e49", font=(base, 12), justify="left",
                               wraplength=960)
        self.rhymes.pack(fill="x", padx=12, pady=(4, 0))
        bar = tk.Frame(root)
        bar.pack(fill="x", padx=12, pady=10)
        for text, verdict in [("Dislike (Esc)", "dislike"), ("Like (Enter)", "like"),
                              ("Superlike (Ctrl+Enter)", "superlike")]:
            tk.Button(bar, text=text, width=22, takefocus=False,
                      command=lambda v=verdict: self.vote(v)).pack(side="left", padx=(0, 8))
        tk.Button(bar, text="Context (Tab)", takefocus=False,
                  command=self.toggle).pack(side="right")
        tk.Button(bar, text="Skip (Down)", takefocus=False,
                  command=self.skip).pack(side="right", padx=(0, 8))
        tk.Button(bar, text="Back (Up)", takefocus=False,
                  command=self.back).pack(side="right", padx=(0, 8))
        self.status = tk.Label(root, anchor="w", fg="#999")
        self.status.pack(fill="x", padx=12)
        self.edit.bind("<Return>", lambda e: self.vote("like") or "break")
        self.edit.bind("<Control-Return>", lambda e: self.vote("superlike") or "break")
        if MAC:
            self.edit.bind("<Command-Return>", lambda e: self.vote("superlike") or "break")
        root.bind("<Escape>", lambda e: self.vote("dislike"))
        self.edit.bind("<Tab>", lambda e: self.toggle() or "break")
        self.edit.bind("<Up>", lambda e: self.back() or "break")
        self.edit.bind("<Down>", lambda e: self.skip() or "break")
        for n in range(1, KEYS + 1):
            self.edit.bind(f"<{'Command' if MAC else 'Alt'}-Key-{n}>",
                           lambda e, n=n: self.use_suggestion(n) or "break")
        self.show()
        self.poll()

    # --- rows -----------------------------------------------------------------------------

    def row(self) -> dict:
        return self.rows[self.order[self.pos]]

    def lines_of(self, sid: str) -> list[str]:
        return self.sections[sid][1]["lines"] if sid in self.sections else []

    def proposed(self, row: dict) -> str:
        """The line as the row proposes it (never an edit you made in another list)."""
        return row.get("rewrite", "") or row["original"]

    def yours(self, row: dict) -> str:
        """Your latest saved version of this line for this list ("" if none)."""
        names = source_names(row) if self.source in (ALL, GARBAGE) else {self.source}
        found = [r for r in self.rows if r is not row and r["id"] == row["id"]
                 and r["line"] == row["line"] and r.get("rewrite")
                 and any(f"{n}: {YOURS}" in r.get("sources", "") for n in names)]
        return found[-1]["rewrite"] if found else (row.get("edit") or "").strip()

    def keep_draft(self) -> None:
        """Remember text typed on the current line but not voted on."""
        if self.paused or self.pos >= len(self.order):
            return
        text = self.edit.get("1.0", "end").strip()
        r = self.row()
        if text and text not in (self.proposed(r), self.yours(r)):
            self.drafts[self.order[self.pos]] = text

    def blank(self, message: str) -> None:
        self.header.config(text=message)
        self.orig.config(state="normal")
        for w in (self.orig, self.edit):
            w.delete("1.0", "end")
        self.orig.config(state="disabled")
        self.rhymes.config(text="")
        self.info.config(text="")
        self.context = False
        self.ctx.pack_forget()

    def show(self) -> None:
        self.paused = False
        if self.pos >= len(self.order):
            self.blank("All lines are done. Close the window.")
            return
        if self.pos == self.boundary and not self.past_pause:
            self.paused = True
            self.edit.focus_set()
            self.blank(f"All {self.boundary} unvetted lines are done. Close the window, or press "
                       f"Enter (or Down) to review the {len(self.order) - self.boundary} lines "
                       f"you already vetted.")
            return
        r = self.row()
        done = sum(bool(self.rows[k].get("verdict")) for k in self.order)
        self.header.config(text=f'{r["artist"]} - {r["title"]}   line {r["line"]}   '
                                f'[{r.get("sources", "")}]   {done}/{len(self.order)} vetted'
                                + (f'   (was: {r["verdict"]})' if r.get("verdict") else ""))
        self.orig.config(state="normal")
        self.orig.delete("1.0", "end")
        self.orig.insert("1.0", r["original"])
        low = r["original"].lower()
        for w in set(self.detector.find(r["original"])):
            for m in re.finditer(rf"(?<![\w']){re.escape(w)}(?![\w'])", low):
                self.orig.tag_add("explicit", f"1.0+{m.start()}c", f"1.0+{m.end()}c")
        self.orig.config(state="disabled")
        self.edit.delete("1.0", "end")
        self.edit.insert("1.0", self.drafts.get(self.order[self.pos]) or self.yours(r)
                         or self.proposed(r))
        self.edit.focus_set()
        self.edit.mark_set("insert", "end-1c")
        self.context = False  # context closes on every line
        self.render_context()
        self.update_info()
        self.load_suggestions()

    def render_context(self) -> None:
        if not self.context:
            self.ctx.pack_forget()
            return
        r = self.row()
        i = int(r["line"]) - 1
        orig = self.lines_of(r["id"])
        new = list(orig)
        if 0 <= i < len(new):
            new[i] = self.edit.get("1.0", "end").strip()
        start = (i // 4) * 4
        self.ctx.config(state="normal")
        self.ctx.delete("1.0", "end")
        for title, lines in [("original", orig), ("rewrite", new)]:
            self.ctx.insert("end", f"{title}\n")
            for j in range(start, min(len(lines), start + 4)):
                self.ctx.insert("end", f"  {lines[j]}\n", ("current",) if j == i else ())
        self.ctx.config(state="disabled", height=11)
        self.ctx.pack(fill="x", padx=12, pady=(8, 0), before=self.info)

    def update_info(self) -> None:
        if self.pos >= len(self.order):
            return
        r, text = self.row(), self.edit.get("1.0", "end").strip()
        left = self.detector.find(text)
        self.info.config(text=f"syllables {syllables(r['original'])} -> {syllables(text)}"
                              + (f"   still explicit: {', '.join(sorted(set(left)))}" if left
                                 else ""))

    def toggle(self) -> None:
        self.context = not self.context
        self.render_context()

    def vote(self, verdict: str) -> None:
        if self.paused:
            if verdict == "like":  # Enter on the pause screen: go on to the vetted lines
                self.go_on()
            return
        if self.pos >= len(self.order):
            return
        r = self.row()
        text = self.edit.get("1.0", "end").strip()
        if self.source == HUMAN:
            if verdict == "dislike":  # nothing to dislike here: Esc skips the line
                self.skip()
                return
            if not text or text == r["original"] or self.detector.find(text):
                self.status.config(text="write a clean version of the line first "
                                        f"({SUGGEST_KEY}+1-9 for rhymes), then Enter")
                return
        if verdict != "dislike" and text and text != self.proposed(r):
            # your own version: a row of its own for this list; the proposal keeps its verdict
            names = source_names(r) if self.source in (ALL, GARBAGE) else {self.source}
            words = set(re.findall(r"[\w']+", text.lower()))
            used = " ".join(sorted(w for w in self.inserted.get(self.order[self.pos], ())
                                   if w.lower() in words))
            mine = add_your_version(self.rows, r, text, verdict, names)
            if mine is None:
                # the same line has another verdict in another list: keep it on this row
                r["edit"], r["verdict"] = text, verdict
                mine = r
            else:
                r["verdict"] = r.get("verdict") or "edited"
            if used:
                mine["suggested"] = used
            saved = f"your version of line {r['line']} ({verdict}, a pair of its own)"
        else:
            r["verdict"] = verdict
            saved = f"line {r['line']} {verdict}"
        try:
            write_sheet(self.rows)
            self.status.config(text=f"saved: {saved}")
        except PermissionError:
            self.status.config(text="could not save: close the sheet in Excel (kept in memory)")
        self.drafts.pop(self.order[self.pos], None)
        self.inserted.pop(self.order[self.pos], None)
        self.history.append(self.pos)
        self.pos += 1
        self.show()

    def go_on(self) -> None:
        self.past_pause = True
        self.show()

    def back(self) -> None:
        if self.history:
            self.keep_draft()
            self.pos = self.history.pop()
            if self.pos < self.boundary:
                self.past_pause = False
            self.show()

    def skip(self) -> None:
        if self.paused:
            self.go_on()
            return
        if self.pos < len(self.order):
            self.keep_draft()
            self.history.append(self.pos)
            self.pos += 1
            self.show()

    # --- suggestions for the explicit words -----------------------------------------------

    def load_suggestions(self) -> None:
        """Queue a request; one worker thread answers the latest request only."""
        self.suggestions = []
        r = self.row()
        current = self.edit.get("1.0", "end").strip()
        self.slots = slots(r["original"], current, self.detector)
        if not self.slots:
            self.rhymes.config(text="")
            return
        self.rhymes.config(text="rhymes for the explicit word: ...")
        i = int(r["line"]) - 1
        lines = self.lines_of(r["id"])
        before = lines[i - 1] if 0 < i <= len(lines) else ""
        self.pending = (self.pos, current, self.slots, before)
        if not getattr(self, "worker", None):
            self.worker = threading.Thread(target=self.work, daemon=True)
            self.worker.start()

    def work(self) -> None:
        from radio_ready.rhymes import suggest_for_word
        from radio_ready.scoring import shared

        try:
            sim = shared("similarity")  # one copy for the app, not one per window
            embed = lambda t: sim.model.encode(t, normalize_embeddings=True)  # noqa: E731
        except Exception as e:  # noqa: BLE001
            self.results.put((None, [], str(e)))
            return
        while True:
            job, self.pending = self.pending, None
            if job is None:
                threading.Event().wait(0.05)
                continue
            pos, current, found_slots, before = job
            tokens, out, err = current.split(" "), [], ""
            try:
                k = per_word(len(found_slots))
                for s in found_slots:
                    template = " ".join(tokens[:s["start"]] + ["<mask>" + s["punct"]]
                                        + tokens[s["end"]:])
                    for c in suggest_for_word(template, s["word"], current, embed, k=k,
                                              before=before):
                        out.append({**c, "slot": s})
                out = out[:KEYS]
            except Exception as e:  # noqa: BLE001 (shown, never fatal)
                err = str(e)
            self.results.put((pos, out, err))  # tkinter is only touched by the main thread

    def poll(self) -> None:
        while not self.results.empty():
            pos, found, err = self.results.get()
            if pos is None or pos == self.pos:
                self.suggestions = found
                if err:
                    self.rhymes.config(text=f"rhymes unavailable: {err}")
                elif not found:
                    self.rhymes.config(text="rhymes for the explicit word: none found")
                else:
                    groups: dict[str, list[str]] = {}
                    for n, s in enumerate(found, start=1):
                        groups.setdefault(s["slot"]["word"], []).append(f"{n}={s['word']}")
                    self.rhymes.config(text=f"{SUGGEST_KEY}+n:   " + "      ".join(
                        f"{w}: " + "  ".join(v) for w, v in groups.items()))
        self.root.after(100, self.poll)

    def use_suggestion(self, n: int) -> None:
        if n > len(self.suggestions):
            return
        s = self.suggestions[n - 1]
        tokens = self.edit.get("1.0", "end").strip().split(" ")
        slot = s["slot"]
        tokens[slot["start"]:slot["end"]] = [s["word"] + slot["punct"]]
        self.inserted.setdefault(self.order[self.pos], set()).add(s["word"])
        self.edit.delete("1.0", "end")
        self.edit.insert("1.0", " ".join(tokens))
        self.update_info()
        self.load_suggestions()  # the slots moved: recompute for the new text


def vetting_queue(rows: list[dict], name: str) -> tuple[list[int], list[int]]:
    """(unvetted, vetted) row indexes a list shows. Every version of a line is an option of its
    own; repeated lines of a section share their versions (radio_ready.pairs.share_repeats),
    so a rewrite is asked once for all occurrences of a line, and not again when the same text
    already has a verdict on any of them. The human list asks a line until any source has an
    accepted version of it, and not while a proposal of it waits for a verdict."""
    if name == GARBAGE:
        mine = [k for k, r in enumerate(rows) if garbage(r)]
    else:
        mine = [k for k, r in enumerate(rows) if (name == ALL or name in source_names(r))
                and not garbage(r)]
    unvetted = [k for k in mine if not rows[k].get("verdict")]
    group = lambda r: (r["id"], line_key(r.get("original", "")))  # noqa: E731
    text = lambda r: (r.get("edit") or "").strip() or r.get("rewrite", "")  # noqa: E731
    if name == HUMAN:
        settled = {(r["id"], r["line"]) for r in rows
                   if r.get("verdict", "").strip().lower() in APPROVE
                   or (r.get("rewrite") and not r.get("verdict", "").strip())}
        accepted = {group(r) for r in rows if r.get("verdict", "").strip().lower() in APPROVE}
        unvetted = [k for k in mine if f"{HUMAN}: {HUMAN_REQUEST}" in rows[k].get("sources", "")
                    and (rows[k]["id"], rows[k]["line"]) not in settled
                    and group(rows[k]) not in accepted]
    judged = {(group(r), text(r)) for r in rows if r.get("verdict")}
    queued, kept = set(), []
    for k in sorted(unvetted, key=lambda k: (rows[k]["id"], int(rows[k]["line"]))):
        key = (group(rows[k]), text(rows[k]))
        if key in judged or key in queued:
            continue
        queued.add(key)
        kept.append(k)
    return kept, [k for k in mine if rows[k].get("verdict")]


def open_source(window, name: str, rows, sections) -> bool:
    """A vetting window over the rows of one source (or all): every unvetted line first, then,
    after a pause screen, the lines already vetted."""
    unvetted, vetted = vetting_queue(rows, name)
    if not unvetted and not vetted:
        return False
    counts, total = word_rarity()
    swaps, n_swaps = substitution_counts(rows)

    def ranked(ks):
        """Most original substitutions first: words the proposals rarely add, then words rare
        in clean lyrics."""
        def score(k):
            text = rows[k].get("edit") or rows[k].get("rewrite", "")
            return (round(uniqueness(rows[k]["original"], text, swaps, n_swaps), 1),
                    uniqueness(rows[k]["original"], text, counts, total))
        return sorted(ks, key=score, reverse=True)

    App(window, rows, ranked(unvetted) + ranked(vetted), sections, ExplicitDetector(), name,
        boundary=len(unvetted))
    return True


class Launcher:
    """Start window: pick a source and vet it. Pairs update by themselves in the background."""

    def __init__(self, root):
        self.root = root
        root.title("RadioReady vetting")
        root.geometry("760x440")
        base = tkfont.nametofont("TkDefaultFont").actual()["family"]
        tk.Label(root, text="Double-click or Enter to vet. New runs and all pairs update by "
                            "themselves (new runs within 30 s while no vetting window is open).",
                 anchor="w", fg="#555").pack(fill="x", padx=12, pady=(10, 4))
        self.list = tk.Listbox(root, font=(base, 12), activestyle="dotbox")
        self.list.pack(fill="both", expand=True, padx=12)
        self.list.bind("<Double-Button-1>", lambda e: self.open())
        self.list.bind("<Return>", lambda e: self.open())
        bar = tk.Frame(root)
        bar.pack(fill="x", padx=12, pady=8)
        tk.Button(bar, text="Vet", width=12, command=self.open).pack(side="left")
        self.status = tk.Label(root, anchor="w", fg="#777", justify="left", wraplength=730)
        self.status.pack(fill="x", padx=12, pady=(0, 10))
        self.busy, self.again, self.waiting, self.windows = False, False, None, 0
        self.done: queue.Queue = queue.Queue()
        self.reload()
        self.runs = self.run_files()
        self.start_update()
        self.root.after(WATCH_MS, self.watch)

    def reload(self) -> None:
        self.sections = load_sections()
        self.rows = load_sheet(self.sections)
        self.refresh()

    def refresh(self) -> None:
        keep = self.selected()
        self.items = [(ALL, f"{ALL}    {self.count(ALL)}")]
        self.items.append((HUMAN, f"{HUMAN} (finish sections yourself)    {self.count(HUMAN)}"))
        self.items.append((GARBAGE, f"{GARBAGE} (no change or nonsense, all sources)    "
                                    f"{self.count(GARBAGE)}"))
        # sources with runs on this machine, then those known only from the shared sheet
        # (outputs/ is not shared: a teammate vets every proposal through the sheet)
        names = list(sources())
        names += sorted({n for r in self.rows for n in source_names(r)} - set(names) - {HUMAN})
        for name in names:
            self.items.append((name, f"{name}    {self.count(name)}"))
        self.list.delete(0, "end")
        for _, label in self.items:
            self.list.insert("end", label)
        names = [n for n, _ in self.items]
        k = names.index(keep) if keep in names else 0
        self.list.selection_set(k)
        self.list.activate(k)
        self.list.focus_set()

    def count(self, name: str) -> str:
        if name == HUMAN:
            asked = {(r["id"], r["line"]) for r in self.rows
                     if f"{HUMAN}: {HUMAN_REQUEST}" in r.get("sources", "")}
            done = {(r["id"], r["line"]) for r in self.rows
                    if r.get("verdict", "").strip().lower() in APPROVE} & asked
            return f"{len(done)}/{len(asked)} lines finished" if asked else "nothing to finish"
        unvetted, vetted = vetting_queue(self.rows, name)
        if not unvetted and not vetted:
            return "new: its lines appear after the update" if self.busy else "nothing to vet"
        return f"{len(vetted)}/{len(vetted) + len(unvetted)} lines vetted"

    def selected(self) -> str | None:
        sel = self.list.curselection() if hasattr(self, "items") else ()
        return self.items[sel[0]][0] if sel else None

    # --- background update --------------------------------------------------------------------

    @staticmethod
    def run_files() -> dict[str, float]:
        """Generation runs on this machine, with their modification times."""
        return {f.name: f.stat().st_mtime for f in paths.GENERATIONS_DIR.glob("*.jsonl")}

    def watch(self) -> None:
        """New or changed runs are picked up by themselves: an update runs as soon as no
        vetting window is open (a window holds its own copy of the sheet)."""
        runs = self.run_files()
        if runs != self.runs and not self.windows:
            self.runs = runs
            self.status.config(text="New runs found ...")
            self.start_update()
        self.root.after(WATCH_MS, self.watch)

    def start_update(self) -> None:
        if self.busy:
            self.again = True  # one more round after this one (new verdicts came in)
            return
        self.busy = True
        self.status.config(text="Updating pairs in the background ...")

        def work():
            out = io.StringIO()
            try:
                with contextlib.redirect_stdout(out):
                    text = update()
            except Exception as e:  # noqa: BLE001 (shown in the window)
                text = f"update failed: {e}"
            self.done.put(text)

        threading.Thread(target=work, daemon=True).start()
        self.root.after(300, self.poll)

    def poll(self) -> None:
        try:
            text = self.done.get_nowait()
        except queue.Empty:
            self.root.after(300, self.poll)
            return
        self.busy = False
        self.reload()
        final = [l for l in text.splitlines() if l.startswith(("final:", "lines:", "update failed",
                                                              "removed old"))]
        self.status.config(text="Up to date. " + " ".join(l.split(" {")[0] for l in final))
        if self.again:
            self.again = False
            self.start_update()
        elif self.waiting:
            name, self.waiting = self.waiting, None
            self.open(name)

    # --- vetting ------------------------------------------------------------------------------

    def open(self, name: str | None = None) -> None:
        name = name or self.selected()
        if not name:
            return
        if self.busy:  # the update writes the sheet: open the window once it is done
            self.waiting = name
            self.status.config(text=f"Updating pairs ... {name} opens when it is done.")
            return
        window = tk.Toplevel(self.root)
        if not open_source(window, name, self.rows, self.sections):
            window.destroy()
            self.status.config(text="Nothing to vet here.")
            return

        self.windows += 1

        def closed():
            window.destroy()
            self.windows -= 1
            self.refresh()
            if not self.windows:  # never while a vetting window can still write the sheet
                self.start_update()

        window.protocol("WM_DELETE_WINDOW", closed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Vet synthetic rewrites line by line.")
    parser.add_argument("source", nargs="?", default=None,
                        help="a source name (default: the start window)")
    args = parser.parse_args()
    root = tk.Tk()
    if args.source:
        sections = load_sections()
        if not open_source(root, args.source, load_sheet(sections), sections):
            raise SystemExit(f"nothing to vet for {args.source}")
    else:
        Launcher(root)
    root.mainloop()


if __name__ == "__main__":
    main()
