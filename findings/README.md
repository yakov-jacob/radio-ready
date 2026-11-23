# Findings

One line per experiment; details in the linked files. Human vetting numbers are from
`annotations/vetting.csv` (share of vetted lines accepted as written = like + superlike; only proposals that passed the source's filters count for it, the share that failed them is its fail rate, new words edited = share of the words a rewrite introduced that the final version replaced (and how many of those with a suggested rhyme), and edits of failed rewrites are human versions; `python -m radio_ready.pairs --stats`).

## Data preparation

| Topic | Result |
| --- | --- |
| Explicit-word detection (MGPHot gold, 16,036 songs) | Learned word list after human vetting: macro-F1 95.4 (precision 88.9, recall 96.2). Surge list at severity >= 2 misses common profanity (shit 1.2, bitch 1.4); at >= 1 it adds everyday words (finger, sheet). Pipeline detector = vetted learned list + Surge words among learned candidates (95.2). |
| Genius sections | 100,169 explicit sections, one per artist (75% rap); cleaning maps homoglyphs, drops scrape junk, filler lines and repeated choruses. |
| Kidz Bop | 300 of 1,353 rows delete the line; most censored words are not radio-explicit (kiss, body, god). Used only as a side-by-side and evaluation set: 119 explicit sections, 50 in the pilot. |
| Official clean versions (Genius) | 421 clean/explicit song pairs; explicit lines are rewritten (643), bleeped (548), muted (224), partial (87) or different (43). 218 fully rewritten sections serve as human references. |
| ParaDeHate | 7,637 usable pairs. The GPT-4o-mini rewrites formalize dialect: AAE proportion 0.225 -> 0.125; 18% of pairs lose >= 0.2. |
| Licenses | `cmudict` Python package is GPL; replaced by reading the BSD-style CMU dictionary file directly (identical output on 29,415 lines). |

## Rewrite generation

| Method | Result |
| --- | --- |
| **free v2** (`free.py`, 7B, section context, stock swaps named) | **Best local method: 69% accepted as written, 7% disliked (29 filter-passed lines; 78% of its 282 proposals fail the filters; 14% of the words it introduced were edited)**, ahead of Claude (59% / 27%) and Gemini (58% / 28%). Chosen for the training data. [free_prompt.md](free_prompt.md) |
| free v3 (mined stock swaps per explicit word) | Stock swaps in rewritten lines: 7 of 20 against 12 of 22 for v2 on the same sections. 46% accepted, 46% edited, 8% disliked (52 lines). |
| Section order for generation runs | The sections file is sorted by artist name; the free v2/v3 runs took it in order (sections 11-817), so artists whose names start with symbols, digits or early letters are over-represented in those pairs. Runs now take a seeded random sample (`free.py --order random`, the default). |
| Lines saturated with profanity | Filter-passed proposals by explicit words in the original line: 1 word 48% accepted as written (791 vetted), 2 words 49% (79), 3+ words 8% (12; half edited). 17,101 of 235,606 explicit lines are dense (>= 3 explicit words, or >= 2 that are >= 25% of the line). `free.py --dense` targets only those lines. |
| One free rewrite per line (free v4, dropped) | 7B rewriting most of each line from its stanza, with a user-written prompt; on the 47 open dense lines: 36 rewritten, 26 mostly, but only 8 kept the end rhyme, and many drifted off-meaning or nearly copied a neighbor line. Dropped as a method; its lines stay in the sheet as data (source free-v4). |
| free v1 | 56% accepted, 0% disliked (16 lines). Naming the stock swaps (v2) cut them from 11 to 4 of 27 lines. |
| sing v2 (`sing.py`, 7B meaning draft + sing pass) on the human-list lines | 62 of 101 lines proposed (55 drafts, 7 sung); the draft wins unless it breaks rhyme or meter. The draft sometimes returns another line of the stanza; candidates must now keep meaning to the original and not copy a line. Not vetted yet. |
| match v4 / sing v3 (`sing.py --scope all`) on all missing lines | 841 lines in 441 sections: match-v4 proposes 411, sing-v3 338 (488 lines with either). Drafts fail mostly because the 7B keeps an explicit word (199) or toxicity rejects a kept "damn"; 126 drift from the original. match-v4 so far: 39% accepted, 23% disliked (31 lines). |
| Noise (14B, noise-vowels) on 238 complementary sections | Sections with disliked commercial rewrites or only stock-swap accepted lines: 144 of 238 pass the filters. Noise source overall: 52% accepted, 34% disliked (61 lines). |
| solid (`rewrite.py`, 7B draft + 14B) | 49% accepted, 19% disliked (116 lines); proposes far fewer lines (8 of 27 on Genius). |
| `generate.py` whole section (v1) | Left explicit words in 37% of pilot sections and changed 27% of clean lines. |
| `generate.py` marked lines (v3, v5) | Clean in 90% of pilot sections, close syllables and rhyme; all `generate.py` pilot runs together: 19% accepted, 71% disliked (259 lines). The 7B standardizes dialect (gon' -> 'll). |
| `generate.py` lines + neighbors (v4) | Freer, but shuffled or invented lines and normalized dialect in neighbor lines. |
| Qwen3-14B (4-bit, embeddings on CPU) | Best meter and rhyme on the pilot, but often returns the explicit line unchanged (7 of 10 clean). |
| Best-of-N with ParaDeHate filters | Used by `generate.py` and `free.py`; the filters remove explicit, toxic and off-meaning candidates. |
| Matching to real clean Genius lines | Retrieved lines rarely keep the meaning (cosine 0.35-0.6); a 7B word-replacement draft does (0.75-0.95). |
| Rhyme swap (`rhyme_swap.py`) | 17 usable of 67 pilot lines; rhyme v2: 25% accepted, 50% disliked (8 lines). |
| Noise styles (explicit words replaced by made-up sounds) | 52% accepted, 34% disliked (61 filter-passed lines; the sense filter drops gibberish). |
| Few-shot full sections (real clean versions as examples) | 1 of 10 pilot sections clean for both 7B and 14B: the models copied the section. Dropped. |
| Commercial windows (3-line windows, Claude) | 39% accepted, 45% disliked (67 lines). |

## Future work

| Topic | Note |
| --- | --- |
| Drop penalty (BLEU-like) | Score rewrites so that only deleting an explicit word, with no flavorful substitute, loses: a brevity penalty `exp(1 - syllables(original) / syllables(rewrite))` when the rewrite is shorter, times the mean over explicit words of (aligned slot filled with a new content word) x (rarity of that word among all proposals, so stock swaps score low). Drop-only rewrites are rare among filter-passed proposals now (0-6% for current sources, 5-8% for the pilots, all disliked when vetted), so it matters most as an evaluation metric and DPO preference signal for the fine-tuned 1.5B. |
| Explicit detector senses | The detector is a word list and stays strict: some vetted lines keep a flagged word in a clean sense (the 'Gram, headline clips, suck at, a name like Dick). Sense-aware detection is left for later. |
