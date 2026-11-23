"""Lyric text utilities: normalization, ad-libs, syllables, stress, end rhyme, rhyme scheme."""

import re
from functools import lru_cache

from radio_ready import cmu

WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)*'?")
TOKEN = re.compile(r"[a-z]+(?:'[a-z]+)*")
HEADER_LINE = re.compile(r"^\s*\[[^\]]*\]\s*$", re.MULTILINE)
PARENTHETICAL = re.compile(r"\(([^()]*)\)")
VOWEL_GROUP = re.compile(r"[aeiouy]+")


# Cyrillic/Greek look-alikes Genius sometimes substitutes for Latin letters.
HOMOGLYPHS = str.maketrans(
    "аеорсухіјАЕОРСХ"
    "ВНКМТѕЅοΟ",
    "aeopcyxijAEOPCXBHKMTsSoO",
)


def normalize(text: str) -> str:
    """Fix homoglyphs, encoding damage and curly punctuation; collapse spaces."""
    text = text.translate(HOMOGLYPHS)
    text = text.replace("�", '"').replace("’", "'").replace("‘", "'")
    text = text.replace("“", '"').replace("”", '"').replace(" ", " ")
    text = text.replace(" ", " ").replace(" ", " ").replace("​", "")
    return re.sub(r"[ \t]+", " ", text).strip()


def words(text: str) -> list[str]:
    return WORD.findall(text)


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens with section headers removed (used by the lexicon and detector)."""
    return TOKEN.findall(HEADER_LINE.sub(" ", str(text).lower()).replace("’", "'"))


def split_adlibs(line: str) -> tuple[str, list[str]]:
    """Remove parenthetical ad-libs, e.g. 'Pour up (Drank)' -> ('Pour up', ['Drank'])."""
    adlibs = [a.strip() for a in PARENTHETICAL.findall(line) if a.strip()]
    main = re.sub(r"\s+", " ", PARENTHETICAL.sub(" ", line))
    main = re.sub(r"\s+([,.!?;:])", r"\1", main).strip(" ,")
    return main, adlibs


SPELLINGS = {"em": "um", "ya": "yuh", "tryna": "trynna", "gon": "gone", "lil": "little"}


def _lookup_word(word: str) -> str:
    """Map slang spellings to dictionary forms: runnin' -> running, 'em -> um."""
    w = word.lower().strip("'")
    if w in SPELLINGS and cmu.phones_for_word(SPELLINGS[w]):
        return SPELLINGS[w]
    if word.lower().endswith("in'") or (w.endswith("in") and not cmu.phones_for_word(w)):
        if cmu.phones_for_word(w + "g"):
            return w + "g"
    return w


@lru_cache(maxsize=200_000)
def word_syllables(word: str) -> int:
    phones = cmu.phones_for_word(_lookup_word(word))
    if phones:
        return cmu.syllable_count(phones[0])
    w = word.lower().strip("'")
    count = len(VOWEL_GROUP.findall(w))
    if w.endswith("e") and not w.endswith(("le", "ee")) and count > 1:
        count -= 1
    return max(count, 1)


def syllables(line: str) -> int:
    return sum(word_syllables(w) for w in words(line))


@lru_cache(maxsize=200_000)
def word_stress(word: str) -> str:
    """CMU stress digits (0 unstressed, 1 primary, 2 secondary); '?' per syllable if unknown."""
    phones = cmu.phones_for_word(_lookup_word(word))
    if phones:
        return cmu.stresses(phones[0])
    return "?" * word_syllables(word)


def stress(line: str) -> str:
    return "".join(word_stress(w) for w in words(line))


@lru_cache(maxsize=200_000)
def rhyme_key(word: str) -> str:
    """Rhyming part (last stressed vowel onward); spelling fallback for unknown words."""
    phones = cmu.phones_for_word(_lookup_word(word))
    if phones:
        return re.sub(r"\d", "", cmu.rhyming_part(phones[0]))
    w = word.lower().strip("'")
    groups = list(VOWEL_GROUP.finditer(w))
    return w[groups[-1].start():] if groups else w


# Line-final words that are sung filler, not the rhyme: "...what I get, yeah".
FILLERS = {"yeah", "yea", "yeh", "yuh", "uh", "uhh", "huh", "oh", "ohh", "ooh", "ay", "ayy",
           "aye", "ey", "hey", "woah", "whoa", "mhm", "mm", "mmm", "hmm", "ha", "haha", "skrrt",
           "brr", "la", "na", "nah"}
# Unstressed words that can trail the rhyming word: "love me" rhymes with "money".
WEAK = {"me", "it", "em", "'em", "them", "him", "her", "us", "a", "the", "of", "to", "for",
        "in", "on", "with", "and"}
# Vowels sung alike (cot-caught merger).
VOWEL_MERGE = {"AO": "AA"}
# Unstressed vowels that are all sung as a schwa.
REDUCED = {"AH", "IH", "UH", "EH"}


def _phones(word: str) -> list[str] | None:
    found = cmu.phones_for_word(_lookup_word(word))
    return found[0].split() if found else None


def _vowel(phone: str) -> str:
    v = re.sub(r"\d", "", phone)
    return VOWEL_MERGE.get(v, v)


def _tail_key(phones: list[str]) -> str:
    """Stressed vowel of the rhyme plus its last vowel if the rhyme has more syllables:
    money/love me -> AH-IY, problems/solve 'em -> AA-AH, heart/dark -> AA."""
    vowels = [p for p in phones if p[-1].isdigit()]
    if not vowels:
        return ""
    stressed = [i for i, p in enumerate(vowels) if p[-1] in "12"]
    tail = vowels[stressed[-1]:] if stressed else vowels[-1:]
    key = _vowel(tail[0])
    if len(tail) == 1:
        return key
    end = _vowel(tail[-1])
    if tail[-1].endswith("0") and end in REDUCED:  # schwa: riches/bitches
        end = "AH"
    return key + "-" + end


def rhyme_words(line: str) -> list[str]:
    """Words of a line without trailing fillers (all words if the line is only filler)."""
    found = [w.lower() for w in words(line)]
    while len(found) > 1 and found[-1].strip("'") in FILLERS:
        found.pop()
    return found


@lru_cache(maxsize=200_000)
def line_rhyme_keys(line: str) -> tuple[str, ...]:
    """Candidate end-rhyme keys of a line; two lines rhyme when they share one.

    One key from the last word, and one from the last stressed word with a weak tail of up
    to two unstressed words ("love me", "faking it", "all of 'em"). Unknown words fall back
    to their last spelled vowel group.
    """
    found = rhyme_words(line)
    if not found:
        return ()
    keys = []
    last = _phones(found[-1])
    if last is None:
        groups = VOWEL_GROUP.findall(found[-1])
        return ("~" + groups[-1],) if groups else ()
    keys.append(_tail_key(last))
    j = len(found) - 1
    while j > 0 and found[j] in WEAK and len(found) - j <= 2:
        j -= 1
    if j < len(found) - 1 and found[j] not in WEAK:
        tail = [_phones(w) for w in found[j:]]
        if all(tail):
            weak = [re.sub(r"\d", "0", p) for t in tail[1:] for p in t]
            keys.append(_tail_key(tail[0] + weak))
    return tuple(dict.fromkeys(k for k in keys if k))


def rhyme_scheme(lines: list[str]) -> str:
    """Letters per line, lines sharing a letter rhyme (union of shared keys); unrhymed lines
    get their own letter."""
    keys = [line_rhyme_keys(l) for l in lines]
    parent = list(range(len(lines)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    first: dict[str, int] = {}
    for i, ks in enumerate(keys):
        for k in ks:
            if k in first:
                parent[find(i)] = find(first[k])
            else:
                first[k] = i
    letters: dict[int, str] = {}
    return "".join(letters.setdefault(find(i), chr(ord("A") + len(letters) % 26))
                   for i in range(len(lines)))


def end_word(line: str) -> str:
    found = rhyme_words(line)
    return found[-1] if found else ""


def end_rhyme_kept(original: str, rewrite: str) -> bool:
    """End word (fillers skipped) kept or a perfect rhyme."""
    a, b = end_word(original), end_word(rewrite)
    return bool(a and b) and (a == b or rhyme_key(a) == rhyme_key(b))


QUOTES = "'\""


def restore_quotes(original: str, rewrite: str) -> str:
    """Put back the quote marks a model rewrite dropped (prompts unchanged; models often lose a
    closing ' or " of the line):
      - a double quote the original has at the start or end of the line and the rewrite lost
        (only when the line's quotes stay balanced as in the original);
      - an opening or closing single quote at the edge of the line, as in the original;
      - a dropped-g apostrophe (fuckin' -> fightin): a rewritten word ending in -in that lines
        up with an original word ending in -in' gets the apostrophe back.
    A rewrite equal to the original, or empty, is returned as is."""
    import difflib

    if not rewrite or rewrite == original:
        return rewrite
    o, r = original.split(" "), rewrite.split(" ")
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(
            None, [w.lower() for w in o], [w.lower() for w in r], autojunk=False).get_opcodes():
        if op != "replace":
            continue
        for k, j in enumerate(range(j1, j2)):
            src = o[i1 + min(k, i2 - i1 - 1)]
            core = r[j].rstrip(".,!?;:\"")
            tail = r[j][len(core):]
            if (re.search(r"[A-Za-z]in'$", src.rstrip(".,!?;:\"")) and len(core) > 3
                    and core.lower().endswith("in") and not core.endswith("'")):
                r[j] = core + "'" + tail
    out = " ".join(r)
    if original.count('"') > out.count('"'):
        fixed = out
        if original.startswith('"') and not fixed.startswith('"'):
            fixed = '"' + fixed
        if (original.count('"') > fixed.count('"') and original.endswith('"')
                and not fixed.endswith('"')):
            fixed += '"'
        if fixed.count('"') % 2 == original.count('"') % 2:  # never leave a quote unbalanced
            out = fixed
    if original.startswith("'") and not out.startswith("'"):
        out = "'" + out
    if (original.endswith("'") and not out.endswith("'")
            and not re.search(r"[A-Za-z]in'$", original)):  # a closing quote, not a dropped g
        out += "'"
    return out
