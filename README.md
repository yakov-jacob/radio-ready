# RadioReady

Clean radio edits of explicit song lyrics that keep the original meaning, syllable count, and
end rhyme.

## Approach

1. **Data**: explicit lyric sections from Genius, rewritten by a local 7B model (`free.py`
   v2, chosen by human vetting) and vetted line by line; sections of Kidz Bop songs as a
   separate side-by-side and evaluation set.
2. **SFT**: LoRA fine-tune Qwen2.5-1.5B-Instruct on ParaDeHate, then on the vetted lyric pairs,
   with optional control tokens for the lines to edit, syllables and rhyme (`sft.py`).
3. **DPO**: preference pairs ranked by the constraint scorers.
4. **Evaluation**: explicit-content removal, semantic similarity, syllable and rhyme accuracy,
   human ratings.

## Layout

```
annotations/  human-edited files (committed)
data/         raw/, artifacts/, processed/ (gitignored)
notebooks/    exploration
outputs/      models and metrics (gitignored)
src/radio_ready/
  download.py         raw data and models
  labels.py           song-level explicit labels
  learned_lexicon.py  learned explicit-word list + human vetting
  classifier.py       section-level explicit classifier
  explicit.py         explicit-word detector used by the pipeline
  kidzbop.py          Kidz Bop song sections and word stats
  genius.py           Genius cleaning, sections, sampling
  aae.py              AAE vernacular proportion
  scoring.py          similarity, toxicity and fluency models
  free.py             synthetic rewrites (7B, section context; the method used)
  sing.py             surgical singable rewrites of the lines that keep sections unpaired
  pairs.py            pairs from generations and the vetting sheet
  vet.py, rhymes.py   human vetting app and its rhyme suggestions (vet.bat)
  paradehate.py       ParaDeHate pairs in the same format
  sft.py              SFT examples with optional control tokens
  commercial.py       copy-paste table for commercial LLMs
  experiments: generate.py, prompts.py, rewrite.py, rhyme_swap.py, noise.py,
               commercial_batch.py, compare.py (see "Other generation methods")
  text.py             normalization, syllables, stress, rhyme
  cmu.py              CMU Pronouncing Dictionary lookups (reads data/raw/cmudict)
  paths.py            file locations
  pipeline.py         runs all steps in order
  llm_judgment/       LLM-judged artifacts, not used by the pipeline
```

## Quick start

From the repository root:

```bash
python -m venv .venv && source .venv/Scripts/activate && pip install -r requirements.txt && pip install -e . && python -m radio_ready.pipeline
```

On Linux/macOS use `source .venv/bin/activate`. The full run downloads ~4 GB and takes roughly
45 minutes; later runs reuse downloaded files.

### On a Mac

Everything except local generation runs on a Mac (CPU), including the vetting app:

```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt && pip install -e . && python -m radio_ready.pipeline
```

- `bitsandbytes` is skipped on macOS (`requirements.txt` marker); only GPU generation needs it.
- The vetting app needs Tk: the python.org installer includes it; with Homebrew Python run
  `brew install python-tk`. Double-click `vet.command` (or `python -m radio_ready.vet`).
  Rhyme suggestions are Cmd+1-9 instead of Alt+1-9; superlike is Ctrl+Enter or Cmd+Enter.
- The generation runs (`outputs/`) are not shared (they contain lyrics). Every proposed line is
  in the committed `annotations/vetting.csv`, so the start window lists each source from the
  sheet and it can be vetted without the run files; pair files of sources whose runs are not
  on the machine are kept, never deleted.
- Local generation (`free.py`, `generate.py`) uses an NVIDIA GPU with 4-bit models; on an Apple
  GPU (MPS) it loads float16 models instead (Qwen2.5-7B needs about 15 GB of unified memory).
- The 1.5B SFT data (`paradehate`, `sft`) is built on the CPU; training code should pick CUDA,
  then MPS, then CPU.
- `annotations/vetting.csv` is one shared file: vet different sources at the same time, or
  pull before and push after a session, to avoid merge conflicts.

## Pipeline

```bash
python -m radio_ready.pipeline
```

Runs every step below in order and stops at the first failure, ending with the pairs rebuilt
from the committed patches (step 11): the vetted pairs are reconstructed without rerunning a
model. Options: `--from STEP`, `--to STEP`, `--only STEP [STEP ...]` (named steps always
run), `--with-classifier`, `--with-noise`, `--with-rhyme` (steps 4, 9 and 10 are skipped by
default; no other step depends on them). Examples:

```bash
python -m radio_ready.pipeline --only pairs                 # reconstruct the pairs only
python -m radio_ready.pipeline --from learned_lexicon       # after editing the vetting file
python -m radio_ready.pipeline --with-rhyme --with-noise    # also new rhyme / noise runs (GPU)
```

| Step | Module | Inputs | Outputs |
| --- | --- | --- | --- |
| 1 | `download` | — | `data/raw/*`, model cache |
| 2 | `labels` | Genius, Spotify, MGPHot | `data/processed/labels/songs.parquet` |
| 3 | `learned_lexicon` | labels, Surge, vetting | `data/artifacts/learned_explicit_words.csv`, `lexicon_metrics.json` |
| 4 | `classifier` | labels | `outputs/explicit_classifier/` |
| 5 | `kidzbop` | Kidz Bop, Genius, step 3 | `data/artifacts/kidzbop_*.csv`, `data/processed/kidzbop/{sections,pilot}.jsonl` |
| 6 | `genius` | Genius, Kidz Bop, step 3 | `data/processed/genius_sections.jsonl` |
| 7 | `aae` | step 6, TwitterAAE | adds `aae_proportion` to step 6 output |
| 7b | `rhyme_vocab` | Genius | `data/artifacts/rhyme_vocab.json` (`rhymes --build`; the vetting app's rhyme suggestions) |
| 8 | `commercial` | steps 5, 6 | `data/commercial/{prompt.txt,commercial_table.csv}` |
| 9 | `noise` (optional) | step 5 | `outputs/generations/qwen2.5-7b-instruct-4bit_pilot_v9-noise-vowels_n5.jsonl` (`generate --style noise-vowels --samples 5`) |
| 10 | `rhyme` (optional) | step 5 | `outputs/generations/rhyme_pilot_rhyme-v2.jsonl` (`rhyme_swap`) |
| 11 | `pairs` | steps 5, 6, `pairs/*.patch.jsonl` | `pairs/<source>.{jsonl,md}`, `final.*`, `lines.*` (`pairs --rebuild`) |

### 2. Labels

Song-level labels joined onto Genius by normalized primary artist + title:

- Spotify `explicit` flag: a song is explicit if any version is flagged. Split by artist
  60/20/20 into train/val/test.
- MGPHot "Explicit Lyrics" score: explicit if ≥ 0.6, clean if 0. These songs form the `gold`
  split and are not trained on.

### 3. Learned lexicon and vetting

Explicit-word list learned from the labels (Fell et al. 2019):
`importance(w) = rate(w | explicit songs) / rate(w | clean songs)`. The lookup size is tuned on
val. `lexicon_metrics.json` reports lookup and logistic regression on test and gold, with the
Surge list (severity ≥ 1 and ≥ 2) and the combined list as comparisons.

`annotations/explicit_word_vetting.csv` holds a `verdict` per word:

| Verdict | Effect |
| --- | --- |
| `not_explicit` | removed before ranking |
| `explicit` | always included |
| `unsure` / blank | none |

`--add-to-vetting N` appends unvetted words from the top N with a blank verdict. Existing rows
are never changed.

### 4. Classifier

DistilRoBERTa trained on one random section per song with the song's Spotify label; a song's
score is the max over its sections; the threshold is tuned on val.

### Explicit-word detection

`explicit.py` is used by steps 5 and 6. A text is explicit if any token is in the word list,
or it contains a censored token (`f*ck`). Word lists (`--detector`):

- `combined` (default): `learned`, plus Surge entries with severity ≥ 1 that are also among the
  learned candidates (`learned_explicit_words.csv`, after vetting)
- `learned`: lookup words from step 3 after vetting, plus words marked `explicit`
- `surge`: Surge AI Obscenity List entries with severity ≥ 1 (`SURGE_MIN_SEVERITY`)

Gold-set lookup results are in `lexicon_metrics.json` (`combined_sev>=*`, `learned_lookup@*`,
`surge_lookup_sev>=*`).

### 5. Kidz Bop songs

A separate dataset, not part of the Genius sample. Uses: 7B rewrites for side-by-side human
ranking against the Kidz Bop version, evaluation of the trained model, and a pilot subset for
testing 7B generation before running it on all Genius sections.

Each Kidz Bop song is matched to Genius (fuzzy title + artist), each Kidz Bop line is located in
the cleaned sections (line similarity ≥ 0.75, `--min-line-match`), and every eligible explicit
section (same rules as step 6) is kept. Sections with an edited line also get `kidz_bop_lines`
(the section with the Kidz Bop lines in place) and `kidz_bop_edits` (position, original, Kidz
Bop line, cut flag, matched bad words). Splits are by song. `pilot` marks a seeded sample of
sections with an edit (`--pilot-size`, default 50).

Outputs: `data/processed/kidzbop/sections.jsonl`, `data/processed/kidzbop/pilot.jsonl`.

Word statistics: per censored word, category, row/occurrence/song/artist counts, cut rate, and
the most common replacements from a word-level diff of original and Kidz Bop lines
(`data/artifacts/kidzbop_words.csv`, `kidzbop_substitutions.csv`).

### 6. Genius sections

Cleaning: section headers become metadata (type, performers); credit headers, inline tags and
scrape artifacts are removed; an exact repeat of 4 consecutive lines inside a section (a chorus
written out twice) is dropped before sections are chosen (`repeats.py`; sections built before
were migrated in place with `python -m radio_ready.repeats --migrate`, which keeps their ids,
records `repeat_lines_dropped` and remaps the vetting sheets, patches and runs); look-alike Cyrillic/Greek letters are mapped to Latin;
parenthetical ad-libs move to `adlibs`; lines without syllables are dropped; repeated sections
are kept once.

Sampling: a section is eligible if its type is verse, chorus, hook, pre-/post-chorus, bridge or
refrain, it has 4–16 lines, and it contains a detected explicit word. Every artist with an
eligible section is kept: one random eligible song per artist, one random eligible section from
it. Kidz Bop songs are excluded.

`data/processed/genius_sections.jsonl` has one JSON object per line, one section per artist.
Example: Tink, "Ride It", Verse 3 (Dej Loaf).

| Field | Meaning | Example |
| --- | --- | --- |
| `id` | section id (hash of artist, title, section index) | `"793b09d47609"` |
| `artist` | Genius artist | `"Tink"` |
| `title` | song title | `"Ride It"` |
| `tag` | Genius genre tag | `"rb"` |
| `year` | release year | `2019` |
| `section_index` | position of the section in the song (0-based) | `6` |
| `n_sections` | sections in the song after cleaning | `8` |
| `section_header` | original header text | `"Verse 3: Dej Loaf"` |
| `section_type` | normalized type | `"verse"` |
| `performers` | names after the colon in the header | `["Dej Loaf"]` |
| `repeats` | times the section appears in the song | `1` |
| `lines` | lyric lines to rewrite, ad-libs removed | `["Put that pussy on Versace plates", "Make me feel some type of way", ...]` |
| `syllables` | syllables per line | `[9, 7, 8, 9, 9, 9, 9, 9]` |
| `stress` | CMU stress per line (1 primary, 2 secondary, 0 unstressed, ? unknown word) | `["111010101", "1111111", "??111101", ...]` |
| `rhyme_scheme` | end-rhyme letter per line: stressed vowel plus the final vowel of multi-syllable rhymes, trailing fillers ("yeah") skipped, weak tails ("love me") joined | `"AAAAAAAA"` |
| `adlibs` | parenthetical ad-libs removed from the lines | `["Right, um", "Type of way", "Wide awake", ...]` |
| `explicit_words` | detected explicit tokens, one per occurrence | `["pussy", "pussy", "fuck", "bitch"]` |
| `eligible_sections_in_song` | eligible sections in this song | `1` |
| `artist_eligible_songs` | artist's songs with an eligible section | `91` |
| `context_before` | up to 8 lines of the previous section | `["Ride all night", "Oh, I wanna ride all night", ...]` |
| `context_after` | up to 8 lines of the next section | `["Let me ride it, let me ride it", "Ride all night", ...]` |
| `aae_proportion` | AAE proportion (step 7; null if too few known tokens) | `0.629` |

<details>
<summary>Full record</summary>

```json
{
  "id": "793b09d47609",
  "artist": "Tink",
  "title": "Ride It",
  "tag": "rb",
  "year": 2019,
  "section_index": 6,
  "n_sections": 8,
  "section_header": "Verse 3: Dej Loaf",
  "section_type": "verse",
  "performers": [
    "Dej Loaf"
  ],
  "repeats": 1,
  "lines": [
    "Put that pussy on Versace plates",
    "Make me feel some type of way",
    "Nightcap, it got him wide awake",
    "Pussy drippin', 'bout to ride his wave",
    "She's so bossy, she intimidate",
    "She so bossy, asked him on a date",
    "Might just fuck him on the interstate",
    "Told him he should give his bitch a break"
  ],
  "syllables": [
    9,
    7,
    8,
    9,
    9,
    9,
    9,
    9
  ],
  "stress": [
    "111010101",
    "1111111",
    "??111101",
    "101011111",
    "111012102",
    "111011101",
    "111110201",
    "111111101"
  ],
  "rhyme_scheme": "AAAAAAAA",
  "adlibs": [
    "Right, um",
    "Type of way",
    "Wide awake",
    "Intimidate",
    "So good",
    "Fuck you on the interstate",
    "Give it a break"
  ],
  "explicit_words": [
    "pussy",
    "pussy",
    "fuck",
    "bitch"
  ],
  "eligible_sections_in_song": 1,
  "context_before": [
    "Ride all night",
    "Oh, I wanna ride all night",
    "Ride all night",
    "Oh, I wanna ride all night"
  ],
  "context_after": [
    "Let me ride it, let me ride it",
    "Ride all night",
    "Oh, I wanna ride all night",
    "Ride all night",
    "Oh, I wanna ride all night"
  ],
  "artist_eligible_songs": 91,
  "aae_proportion": 0.629
}
```

</details>

`data/processed/kidzbop/sections.jsonl` uses the same fields plus `kidz_bop_artist`,
`kidz_bop_song`, `has_kidz_bop_edit`, `kidz_bop_lines`, `kidz_bop_edits`, `split`, `pilot`.

### 7. AAE proportion

Proportion of tokens from the African American English topic of the TwitterAAE model
(Blodgett et al. 2016). `aae.vernacular_shift(scorer, original, rewrite)` returns the
proportion before and after a rewrite.

## Rewrite generation

The synthetic pairs come from **`free.py` version 2** (Qwen2.5-7B-Instruct, 4-bit), chosen by
human vetting (below). One request per 4-line block with a flagged line (a line with a detected
explicit word): the block is shown as context, flagged lines are marked with their censored
words, syllable count and the end words they rhyme with, and may be reworded freely as long as
the censored words are gone and meaning, voice, slang, syllables and end rhyme are kept.
Unflagged lines are always restored as written. Version 2 also names the stock swaps of the
vetting sheet (shit -> stuff, bitch -> girl, fuck -> mess, ...) and asks for a fresh rewrite
instead. Per block: one greedy answer and 4 samples (temperature 1.0, top-p 0.9); a candidate is
out if it keeps a detected or disguised explicit word, is toxic (>= 0.9), unchanged, or brings
in a non-word; the rest are ranked by syllables (within one), end rhyme, then similarity x
relative CoLA fluency. Details: `findings/free_prompt.md`.

**Version 3** (default) mines the stock swaps automatically instead of taking them from the
vetting sheet: every replacement of an explicit word in every generation run, the vetting
sheet, the Kidz Bop substitutions and the official clean versions is counted per explicit word
(inflections merged). A one-word replacement used for 3+ different lines of a word is a stock
swap of that word (shit -> stuff, thing, nonsense, trash; bitch -> girl, chick, lady; nigga ->
homie, folks); one that is a stock swap of 3+ words (girl, man, thing, hell, ...) or on the v2
list is generic for every line. The prompt names the stock swaps of the words flagged in the
block; a rewrite that brings one in ranks below those that do not. On the same 10 sections,
rewritten lines with a stock swap: v2 12 of 22, v3 7 of 20.

```bash
python -m radio_ready.free --limit 100    # 100 new sections, a random sample
```

The sections file is sorted by artist name (one section per artist), and the earlier free-v2 and
free-v3 runs took it in file order, so they over-sample artists whose names start with symbols,
digits or early letters. Runs now shuffle the sections with a new seed each time, from the clock (`--order random`; the seed
is printed and saved in every output record, `--seed N` repeats it), leave out
artist names that start with a symbol or digit (`--symbol-names` keeps them), and always skip
sections any run of any method already covers, so every run adds new lines. The default output
name has the date and time, so a run never overwrites another.

Each run writes `outputs/generations/<name>.jsonl`; the vetting app lists all `free-v2` runs as
one source.

**Human vetting** (`annotations/vetting.csv`, `python -m radio_ready.pairs --stats`). A source
is credited only with proposals that passed its filters; its fail rate is the share of the lines
it was asked to rewrite that failed (filters, nonsense, no usable answer, or the line back
unchanged). New words edited: of the words a rewrite changed or added against the original, the
share the final version replaced (accepted and edited lines; a line accepted as written counts
0). By rhyme suggestion: the share of those replaced with a word from the vetting app's rhyme
bar (recorded in the sheet's `suggested` column since it was added; for earlier rows, inferred:
a clean word that perfectly rhymes with the original's explicit word, under the bar's rules).
Edits of rewrites that failed the filters are human versions, with the lines written in the
vetting app, not a source's result.

| Source | Failed (of lines asked) | Vetted lines | Accepted as written (like + superlike) | Edited | Disliked | New words edited | of those, by rhyme suggestion |
| --- | --- | --- | --- | --- | --- | --- | --- |
| free-v4 | 89% of 72 | 5 | 80% | 0% | 20% | 0% of 11 | - |
| free-v2 | 81% of 332 | 33 | 64% | 21% | 15% | 12% of 43 | 0% of 5 |
| Claude (commercial) | - (not filtered) | 225 | 59% | 14% | 27% | 15% of 210 | 0% of 32 |
| free-v1 | 66% of 47 | 16 | 56% | 44% | 0% | 38% of 24 | 0% of 9 |
| Gemini (commercial) | 1% of 252 | 151 | 54% | 19% | 27% | 20% of 137 | 7% of 27 |
| noise | 49% of 403 | 62 | 52% | 15% | 34% | 19% of 48 | 0% of 9 |
| solid-v1 | 69% of 1047 | 116 | 49% | 32% | 19% | 27% of 123 | 3% of 33 |
| free-v3 | 71% of 1928 | 52 | 46% | 46% | 8% | 31% of 65 | 0% of 20 |
| sing-v3 | 84% of 824 | 11 | 45% | 9% | 45% | 12% of 8 | 0% of 1 |
| sing-v2 | 65% of 88 | 31 | 42% | 42% | 16% | 32% of 40 | 0% of 13 |
| Claude windows (commercial) | - (not filtered) | 67 | 39% | 16% | 45% | 18% of 51 | 0% of 9 |
| match / sing pilot | 4% of 24 | 13 | 38% | 54% | 8% | 27% of 22 | 0% of 6 |
| match-v4 | 79% of 825 | 44 | 36% | 41% | 23% | 41% of 54 | 0% of 22 |
| guided-v1 | 67% of 24 | 8 | 25% | 75% | 0% | 69% of 13 | 0% of 9 |
| rhyme-v2 | 88% of 67 | 8 | 25% | 25% | 50% | 40% of 5 | 0% of 2 |
| generate.py pilot runs | 5% of 370 | 259 | 19% | 9% | 71% | 18% of 119 | 0% of 21 |

Human versions: 598 (402 lines written in the vetting app, 196 edits of rewrites that failed the filters). Rewrites that failed the filters count only in a source's fail rate (475 of them were judged as written).

### Other generation methods (experiments, not used for the training data)

Kept for the record; each is a source in the vetting app. Findings: `findings/README.md`.

- `generate.py`: prompt styles v1-v9 (whole section, annotated section, one line or 4-line stanza
  with context, freer rewording, noise words in place of explicit words), optional best-of-N
  with the ParaDeHate filters. Qwen3-14B runs with `--cpu-embeddings` on an 8 GB GPU.
- `rewrite.py` (solid): 7B word-replacement draft, Qwen3-14B for hard lines.
- `rhyme_swap.py`: explicit word swapped for a clean rhyme (`rhymes.py`), then reworded by the 7B.
- `commercial_batch.py`: batches of new lines or 3-line windows for Claude / Gemini.
- `compare.py`: automatic side-by-side scores of any runs (explicit left, syllables and rhyme on
  changed lines, clean lines kept, slang kept, LaBSE, AAE delta) and a blinded ranking sheet.

### Synthetic pairs

`pairs.py` turns each source into pairs. Sources: the `solid_*` rewrite runs, the rhyme swaps
(`rhyme_*`; they need your approval), the commercial models (`commercial_*`, from
`python -m radio_ready.commercial --export`; they need your approval and may use the reserved
commercial-evaluation sections). In `pairs/`:

| File | Committed | Contents |
| --- | --- | --- |
| `<source>.patch.jsonl` | yes | ids, split, how the rewrite was made, your verdicts, and a word-level diff of each changed line (`linediff.py`); no lyrics |
| `<source>.jsonl` | no (lyrics) | full pairs: `original`, `rewrite`, `changed`, `split`, model and prompt version; no training features |
| `<source>.md` | no (lyrics) | side-by-side view |
| `final.*` | patch yes | every source combined; identical pairs (same section, same rewrite) kept once |
| `lines.*` | patch yes | one pair per line you liked or superliked (your edit if you made one), with its section as context and your `verdict` as a plain field; disliked lines never appear, and a liked line counts even when its section version is out |
| `share.csv` | no (lyrics) | one simple file to share by hand: every section pair (whole section per cell, `changed_lines`, verdicts) and every line pair (`kind` = section / line), rewritten on every update; opens in Excel |

Every version of a line you accepted is a pair of its own: the proposal when you liked or
superliked it (or it passed the filters, needs no approval and you have not vetted it), and
each of your own versions of it that you liked or superliked. A section gives one pair with the
first accepted version of every line, plus one pair per further accepted version of a line.
Repeated lines (the same explicit line more than once in a section) share their versions: a
version accepted for any occurrence counts for all of them, every pair uses one version for all
occurrences, and each further accepted version is a complete-section pair of its own. The
vetting app asks a repeated line once. On the current sheet this pairs 19 more sections and
skips 275 repeat lines in the vetting queue.
A section is left out when one of its lines has no accepted version and was disliked, or needs
approval; and when something explicit is left. Identical pairs (same section, same lines) are
kept once, within a source and in `final`; different versions are never merged.
With the section records built (`download`, `kidzbop`, `genius`), the full pairs come back from
the committed patches:

```bash
python -m radio_ready.pairs --rebuild
python -m radio_ready.pairs --update   # every source, then pairs/final and pairs/lines
```

### Human vetting

Double-click `vet.bat`: a start window lists every source with how many of its lines are
vetted, plus "All sources". Every run in `outputs/generations` is listed by itself, and runs
of one method are one source whatever sections or batch they cover (`solid-v1`, `free-v2`,
`guided-v1`, `commercial-claude` for the pasted answers and batches, `commercial-windows-claude`
for the 3-line-window batches). Vet opens one: all unvetted lines first, then a pause screen
where you can close the window or press Enter to go on to the lines already vetted. Nothing is
exported by hand: when the start window opens and after each vetting window closes, new runs
are added and every source's pairs, `pairs/final` and `pairs/lines` are rebuilt in the
background; a vetting window asked for meanwhile opens when that is done.

One line at a time: the original with its explicit words highlighted and the rewrite in an
edit field. Edit in place if needed, then Enter (like), Ctrl+Enter (superlike) or Esc (dislike);
Up/Down go back or skip; Tab shows the 4-line stanza, original and rewritten (it closes on
every new line). Alt+1-9 puts a suggestion in place of an explicit word: clean words that
rhyme with the explicit word itself (perfect, slant or assonance: bitch -> cinch, kids, wish),
with its syllable count, never a sound-alike of it or of its last sounds, never a word on the
Surge list or Kidz Bop's profanity list (piss, crap) (`rhymes.py`: CMU rhymes of words common
in clean lyrics, scored by masked-LM fit in the slot and meaning, then picked to differ from
each other in meaning and sound; built by the pipeline's `rhyme_vocab` step). Every explicit
word of the line gets suggestions, numbered in order: 9 for one word, 4 each for two, 3 each
for three, fewer as there are more. Distinctive rewrites come first,
generic ones ("stuff", "things") last.

**Finishing sections yourself (`human` list).** A section is not paired when one of its explicit
lines has no accepted version. The `human` list at the top of the start window asks for exactly
those lines: the section has accepted lines, but this line has no accepted version in any source
and no proposal waiting for your verdict (every proposal was disliked, or no model rewrote it).
You write the clean line (Alt+1-9 for rhymes; Enter only saves a clean, changed line; Esc
skips). `python -m radio_ready.sing` proposes surgical, singable rewrites of the same lines
(source `sing-v2`): a 7B meaning draft that replaces only the explicit words, then sampled
rewrites that make it sing (syllables and rhyme partners given); a candidate must keep the
meaning (cosine >= 0.70 to the draft and to the original) and not copy another line. Vet them in
the `sing-v2` list first; the `human` list then shows only lines with no accepted or pending
version. A section that no source pairs by itself, and whose explicit lines all have an
accepted version somewhere (yours included), is a pair of the `human` source. First run: 101
lines in 53 sections, 62 with a proposal (55 drafts, 7 sung rewrites) in 6 minutes.

`python -m radio_ready.sing --scope all` (the default) covers every explicit line still missing
in any generated section (no accepted version, nothing pending, including lines whose every
candidate failed the filters: 841 lines in 441 sections) and writes two sources: `match-v4`, the
meaning draft alone (the most surgical edit), and `sing-v3`, the best sung rewrite, so both are
offered instead of one pick per line. Lines an earlier `sing-v2` run drafted reuse its candidates;
a text you disliked for a line is never proposed again. Meaning: >= 0.70 to the draft and >= 0.60
to the original line.

**Noise versions for a complementary set.** `data/processed/noise_complement.jsonl`: the 238
sections whose commercial rewrites you disliked (63) or whose accepted lines all use a stock swap
(190), without the sections reserved for commercial evaluation (never paired) or with a noise run
already; rewritten by Qwen3-14B with `generate.py --style noise-vowels --samples 5` (source
`noise`).

Rhyme suggestions also cover rap spellings missing from the CMU dictionary (-a, -as, -az for
-er, -ers: playa, killa, nigga): their pronunciation comes from the -er form. Explicit words,
Surge and Kidz Bop profanity, and sound-alikes of the replaced word are never suggested.

All verdicts go to one sheet, `annotations/vetting.csv` (committed, no lyrics: rewrites and
edits are diffs against the original line). It has one row per line and proposed rewrite, so a
rewrite several sources proposed (the 7B/14B run, a commercial model) is vetted
once and counts for all of them; `sources` lists who proposed it. A line you change is a row of
its own (status `your version`, for the list you vetted it in): the proposal keeps its own
verdict, other lists still show the proposal, and each different version you write is a
separate pair. The same version written in two lists is one row for both. (Older sheets kept
one edit per row, so an edit in a second list replaced the first; those edits are split into
rows of their own on load, and one edit lost that way came back from the sheet's history.)

### Commercial LLMs (copy-paste)

The pipeline's `commercial` step writes one generic prompt, `data/commercial/prompt.txt`, and
one table, `data/commercial/commercial_table.csv`, with a row per section: every Kidz Bop pilot
section plus 50 Genius sections chosen by a hash of the section id (the local model never
rewrites those Genius sections). The prompt does not mark lines, syllables or rhyme: the model
decides what radio would censor.

To fill it:
1. paste `prompt.txt` at the start of a chat (or in front of each section),
2. for each row, copy the `section` cell into the chat,
3. paste the answer into that model's column (`output_chatgpt`, `output_claude`,
   `output_gemini`; rename a column to the exact model, e.g. `output_gemini-2.5-pro`, or add
   one).

The prompt asks for the whole section on one line with lines separated by ` / `, so each
answer fits in one cell. No import step: `compare` reads the filled cells directly and reports
answers it cannot use (wrong number of lines). Re-running the step only adds new rows and model
columns and never changes filled cells; close the file in Excel first. Options:
`python -m radio_ready.commercial --genius-sections 100 --models gpt-5`.

Batches of new lines or 3-line windows, one chat message each (`commercial_batch.py`;
`--items N`, `--version 2` for windows, `--export` after pasting the answers into
`batch_NNN_<model>.txt`) become the sources `commercial-<model>` and `commercial-windows-<model>`.

## Training data

**ParaDeHate** (first SFT stage). `python -m radio_ready.download paradehate`, then:

```bash
python -m radio_ready.paradehate
```

Writes `data/processed/paradehate/pairs.jsonl` in the pair format of `pairs.py` (one-line
`original` and `rewrite`, `changed = [0]`, `source = "paradehate"`), plus `explicit_left`,
`aae_original` and `aae_rewrite`. Of 8,276 pairs, 7,637 are kept (581 duplicate originals, 18
empty or unchanged, 40 with a detected explicit word in the rewrite dropped; `--keep-explicit`
keeps those); split 80/10/10 by a hash of the original: 6,094 / 782 / 761. The GPT-4o-mini
rewrites formalize the text: the AAE proportion drops from 0.225 to 0.125 on average and by 0.2
or more in 18% of pairs.

**SFT examples** from any pair files (default `pairs/final.jsonl` and the ParaDeHate pairs):

```bash
python -m radio_ready.sft                              # no control tokens
python -m radio_ready.sft --controls edit syl rhyme    # with control tokens
```

Writes `data/processed/sft/<name>.jsonl` (`id`, `source`, `split`, `controls`, `prompt`,
`completion`) and `<name>.tokens.json`, the control tokens to add to the tokenizer. The prompt
is an instruction and the original lines, each preceded by its control tokens; the completion is
the rewritten lines (unchanged lines copied). Control values come from the original lines, so
they are known at inference time:

| Control | Token | Value |
| --- | --- | --- |
| `edit` | `<edit>` | line has a detected explicit word |
| `syl` | `<syl:N>` | syllables of the line (max 30) |
| `rhyme` | `<rhyme:X>` | the line's letter in the section's rhyme scheme |
| `end` | `<end:KEY>` | end-rhyme sound (open vocabulary, not added to the tokenizer) |
| `stress` | `<stress:0101>` | CMU stress pattern (open vocabulary) |

ParaDeHate texts are one line, so they get only `edit` and `syl`. The token spelling and the
instructions are constants at the top of `sft.py`.

```
Rewrite these lyrics as a clean radio edit. Keep the line count, meaning, syllables, rhyme, slang and dialect.
<edit> <syl:12> <rhyme:A> You'd better jump ship, this bastard is sinking fast
<syl:13> <rhyme:A> There's nothing to save here, our worth has all but passed
```

## Scoring models

| Model | Use | Threshold |
| --- | --- | --- |
| `sentence-transformers/all-distilroberta-v1` | similarity filter | cosine ≥ 0.70 (ParaDeHate) |
| `unitary/unbiased-toxic-roberta` | toxicity filter | rewrite must score < 0.9 (ParaDeHate) |
| `cointegrated/roberta-large-cola-krishna2020` | fluency, candidate ranking (drop relative to the original line) | — |
| `sentence-transformers/LaBSE` | meaning-preservation metric | — |
| `s-nlp/roberta_toxicity_classifier` | clean-output metric | — |

## LLM-judged artifacts (not in the pipeline)

`src/radio_ready/llm_judgment/` contains artifacts based on Claude's judgment. No pipeline step
imports them.

- `claude_word_tiers.py`: strong / mild / kb_only tiers for Kidz Bop words, with word sets
  written by Claude. Output: `data/artifacts/llm_judgment/claude_word_tiers.csv`.

## Licenses

The code in this repository is MIT-licensed ([LICENSE](LICENSE)). Data, models and third-party
code keep their own licenses:

### Datasets

| Dataset | License |
| --- | --- |
| [ParaDetox](https://huggingface.co/datasets/s-nlp/paradetox) | OpenRAIL++ |
| [ParaDeHate](https://huggingface.co/datasets/ScaDSAI/ParaDeHate) | no license stated on the dataset card (research use) |
| [The Pudding data (Kidz Bop)](https://github.com/the-pudding/data/tree/master/kidz-bop) | MIT |
| [Genius lyrics (cleaned)](https://huggingface.co/datasets/theelderemo/genius-lyrics-cleaned) | MIT (dataset); lyrics are copyrighted by their rights holders |
| [Spotify tracks](https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset) | BSD |
| [MGPHot](https://zenodo.org/records/15372063) | CC BY-NC-ND 4.0 |
| [Surge AI Obscenity List](https://huggingface.co/datasets/mmathys/profanity) | MIT |
| [TwitterAAE model](https://github.com/slanglab/twitteraae) | MIT |
| [CMU Pronouncing Dictionary](https://github.com/cmusphinx/cmudict) (`cmudict.dict`, pinned commit) | BSD-2-Clause-style, © 1993-2015 Carnegie Mellon University |

### Models

| Model | License |
| --- | --- |
| `sentence-transformers/all-distilroberta-v1` | Apache-2.0 |
| `unitary/unbiased-toxic-roberta` | Apache-2.0 |
| `sentence-transformers/LaBSE` | Apache-2.0 |
| `s-nlp/roberta_toxicity_classifier` | OpenRAIL++ |
| `distilroberta-base` | Apache-2.0 |
| `Qwen/Qwen2.5-1.5B-Instruct`, `Qwen/Qwen2.5-7B-Instruct`, `Qwen/Qwen3-14B` | Apache-2.0 |
| `cointegrated/roberta-large-cola-krishna2020` | see model card |

### Code

| Component | License |
| --- | --- |
| `aae.py` inference, adapted from `slanglab/twitteraae` `predict.py` | MIT, © 2019 Su Lin Blodgett and Brendan O'Connor (notice in file) |

## References

- Blodgett, Green, O'Connor. Demographic Dialectal Variation in Social Media: A Case Study of
  African-American English. EMNLP 2016.
- Fell, Cabrio, Corazza, Gandon. Comparing Automated Methods to Detect Explicit Content in Song
  Lyrics. RANLP 2019.
- Logacheva et al. ParaDetox: Detoxification with Parallel Data. ACL 2022.
- Yuan et al. LLM in the Loop: Creating the ParaDeHate Dataset for Hate Speech
  Detoxification. Findings of IJCNLP-AACL 2025.
