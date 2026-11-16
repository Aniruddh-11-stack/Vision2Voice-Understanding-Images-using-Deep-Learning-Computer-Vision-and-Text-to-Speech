"""Turns raw VGG16 + LSTM captions into clean, grammatical spoken English.

The Flickr8k-trained LSTM drops articles ("man is standing on the side of bus"), repeats
words, stops mid-phrase, and per-crop captions often repeat the main one. Two layers fix this:

1. Deterministic rules (always on): stutters and repeated phrases, dangling words,
   missing articles at the start and after prepositions/verbs, a/an, capitals, punctuation;
   object counts and natural lists; de-duplication of crop captions.
2. Optional neural grammar correction (T5), accepted only when the edit is small, so it
   can't add or remove content.
"""
from __future__ import annotations

import os
import re
from collections import Counter
from difflib import SequenceMatcher
from functools import lru_cache

DANGLING = {"a", "an", "the", "with", "and", "of", "in", "on", "at", "to", "for", "by", "next",
            "near", "from", "while", "that", "is", "are", "its", "his", "her", "their", "into",
            "onto", "behind", "under", "over", "wearing", "holding", "front"}
AN_EXCEPTIONS = ("uni", "use", "usu", "uti", "eu", "one", "once", "ubiq", "ura")
A_EXCEPTIONS = ("hour", "honest", "honou", "heir")
NUM_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
IRREGULAR = {"person": "people", "man": "men", "woman": "women", "child": "children",
             "mouse": "mice", "foot": "feet", "tooth": "teeth", "sheep": "sheep", "knife": "knives",
             "wine glass": "wine glasses", "skis": "skis", "scissors": "scissors", "bus": "buses"}
SYNONYMS = {
    "person": {"person", "people", "man", "men", "woman", "women", "boy", "boys", "girl", "girls",
               "child", "children", "kid", "kids", "lady", "guy", "player", "players", "baby", "crowd"},
    "dining table": {"table", "dining table"}, "cell phone": {"phone", "cellphone", "cell phone"},
    "couch": {"couch", "sofa"}, "tv": {"tv", "television"},
    "motorcycle": {"motorcycle", "motorbike", "bike"}, "bicycle": {"bicycle", "bike", "cyclist"},
    "car": {"car", "cars", "vehicle"}, "sports ball": {"ball", "soccer", "football"},
}

DETERMINERS = {"a", "an", "the", "this", "that", "these", "those", "some", "many", "several", "few",
               "his", "her", "their", "its", "my", "our", "your", "another", "each", "every", "no",
               "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
               "lots", "all", "both", "other", "much", "more"}
PRONOUNS = {"it", "he", "she", "they", "him", "them", "we", "i", "you", "there", "something", "someone"}
# words after which a bare singular noun phrase needs an article
ARTICLE_TRIGGERS = {"of", "on", "in", "at", "with", "near", "behind", "under", "over", "into", "onto",
                    "through", "across", "beside", "from", "by", "along", "around", "against", "toward",
                    "towards", "inside", "outside", "underneath", "is", "are", "to",
                    # transitive verbs (both forms the Flickr8k LSTM produces)
                    "wearing", "holding", "riding", "carrying", "throwing", "catching", "pushing", "pulling",
                    "eating", "kicking", "chasing", "climbing", "watching", "hitting", "swinging", "grabbing",
                    "wears", "holds", "rides", "carries", "throws", "catches", "pushes", "pulls", "eats",
                    "kicks", "chases", "climbs", "watches", "hits", "swings", "grabs", "tackles", "does",
                    "has", "drives", "jumps", "leaps", "crosses", "enters", "leaves", "reads", "drinks"}
# nouns that read naturally without an article ("in water", "black hair", "riding gear")
MASS_NOUNS = {"water", "grass", "snow", "sand", "air", "midair", "mud", "rain", "traffic", "background",
              "dirt", "ice", "sunlight", "shade", "hair", "paint", "gear", "clothing", "clothes", "jeans",
              "shorts", "pants", "sunglasses", "glasses", "makeup", "denim", "leather", "costume", "uniform",
              "music", "surf", "foliage", "fog", "smoke", "light", "food", "equipment", "lot", "front",
              "line", "unison", "fall", "winter", "summer", "spring", "autumn", "night", "daytime", "action",
              "motion", "place", "time", "order", "public", "play", "work", "school", "home", "town",
              "camera", "together", "down", "up", "out", "off", "hand", "each", "it", "the"}
SPORTS = {"soccer", "football", "basketball", "baseball", "tennis", "hockey", "rugby", "volleyball",
          "lacrosse", "golf", "cricket", "chess", "guitar", "music"}
SPORT_NOUNS = {"ball", "field", "player", "game", "match", "team", "goal", "net", "court", "jersey", "stick",
               "bat", "helmet", "uniform", "pitch", "stadium", "board", "racket", "club", "game"}
COLOURS = {"black", "white", "red", "blue", "green", "yellow", "orange", "pink", "purple", "brown",
           "gray", "grey", "colorful", "colourful", "dark", "bright", "striped", "plaid"}
ADJECTIVES = COLOURS | {"grassy", "wooded", "rocky", "sandy", "snowy", "muddy", "dirt", "shallow", "deep",
                        "tall", "high", "small", "large", "big", "little", "young", "old", "wet", "dry",
                        "busy", "crowded", "empty", "green", "long", "short", "wooden", "metal", "stone",
                        "brick", "city", "country", "open", "steep", "narrow", "wide", "crowd", "aerodynamic"}
VERB_HINT = re.compile(r"(ing|ed)$")


# ---------- rule layer ----------
def _collapse_repeats(words: list[str]) -> list[str]:
    out: list[str] = []
    for w in words:
        if out and w.lower() == out[-1].lower():
            continue
        out.append(w)
    changed = True
    while changed:  # "on a bench on a bench" -> "on a bench"
        changed = False
        for n in range(6, 1, -1):
            for i in range(len(out) - 2 * n + 1):
                if [w.lower() for w in out[i:i + n]] == [w.lower() for w in out[i + n:i + 2 * n]]:
                    del out[i + n:i + 2 * n]
                    changed = True
                    break
            if changed:
                break
    return out


def fix_articles(text: str) -> str:
    def repl(m: re.Match) -> str:
        art, word = m.group(1), m.group(2)
        lw = word.lower()
        vowel = lw[0] in "aeiou"
        if lw.startswith(AN_EXCEPTIONS):
            vowel = False
        if lw.startswith(A_EXCEPTIONS):
            vowel = True
        new = "an" if vowel else "a"
        return (new.capitalize() if art[0].isupper() else new) + " " + word
    return re.sub(r"\b(a|an|A|An)\s+([A-Za-z][\w'-]*)", repl, text)


def _is_plural(word: str) -> bool:
    w = word.lower()
    return w in {"people", "men", "women", "children", "feet", "teeth"} or (
        w.endswith("s") and not w.endswith(("ss", "us", "is", "ous")) and len(w) > 3)


def _needs_article(prev: str, w: str, nxt: str) -> bool:
    if prev not in ARTICLE_TRIGGERS or w in DETERMINERS | PRONOUNS | ARTICLE_TRIGGERS or w.isdigit():
        return False
    if _is_plural(w) or w.endswith("ing"):            # "with dogs", "in fishing"
        return False
    if prev in {"is", "are", "to"} and (VERB_HINT.search(w) or w in ADJECTIVES | MASS_NOUNS | {"very", "not"}):
        return False                                   # "is running", "is wet", "to play"
    if prev in {"is", "are"} and nxt in ARTICLE_TRIGGERS | {"", "and"}:
        return False                                   # "is outside", "is ready"
    if w in SPORTS:
        return nxt in SPORT_NOUNS                      # "kicks a soccer ball" but "plays soccer"
    if w in ADJECTIVES:                                # "in a grassy field" but "with black hair"
        return bool(nxt) and nxt not in MASS_NOUNS | ARTICLE_TRIGGERS | {"and", "or"} and not _is_plural(nxt)
    if w in MASS_NOUNS:
        return False
    if nxt in MASS_NOUNS and nxt not in {"line", "front"}:  # unknown adjective + mass noun: "aerodynamic gear"
        return False
    return True


def insert_articles(words: list[str]) -> list[str]:
    """Adds 'a' before bare singular noun phrases after prepositions and transitive verbs."""
    out: list[str] = []
    for i, w in enumerate(words):
        prev = out[-1].lower() if out else ""
        j = i + 1
        while w.lower() in ADJECTIVES and j + 1 < len(words) and words[j].lower() in {"and", "or"} \
                and words[j + 1].lower() in ADJECTIVES:
            j += 2                                     # "red and white helmet": look past the adjectives
        nxt = words[j].lower() if j < len(words) else ""
        if _needs_article(prev, w.lower(), nxt):
            out.append("a")
        out.append(w)
    return out


NO_ARTICLE_START = DETERMINERS | PRONOUNS | {"people", "men", "women", "children", "kids", "group"}


def add_leading_article(words: list[str]) -> list[str]:
    """'man is wearing a hat' -> 'a man is wearing a hat'."""
    if not words:
        return words
    first = words[0].lower()
    if first in NO_ARTICLE_START or first.isdigit():
        return words
    plural = _is_plural(first) or any(w.lower() in {"are", "were"} for w in words[:3])
    return words if plural else ["a"] + words


def clean_caption(raw: str) -> str:
    """Model caption -> tidy phrase, no final period."""
    t = (raw or "").strip()
    t = re.sub(r"\b(start|end|startseq|endseq)\b", " ", t, flags=re.I) if t else t
    t = re.sub(r"[^\w\s'’,-]", " ", t)
    t = re.sub(r"\b(\w+) and \1\b", r"\1", t)        # "helmet and helmet" -> "helmet"
    words = _collapse_repeats(t.split())
    while words and words[-1].lower().strip(",") in DANGLING:
        words.pop()
    while words and words[0].lower() in {"and", "with", "of", "is", "are"}:
        words.pop(0)
    words = add_leading_article(words)
    words = insert_articles(words)
    words = _collapse_repeats(words)
    t = " ".join(words).strip(" ,")
    return fix_articles(t)


def sentence(text: str) -> str:
    t = re.sub(r"\s{2,}", " ", text.strip())
    if not t:
        return ""
    t = t[0].upper() + t[1:]
    return t if t[-1] in ".!?”\"" else t + "."


def pluralize(noun: str) -> str:
    if noun in IRREGULAR:
        return IRREGULAR[noun]
    head, _, last = noun.rpartition(" ")
    pre = head + " " if head else ""
    if last in IRREGULAR:
        return pre + IRREGULAR[last]
    if re.search(r"(s|x|z|ch|sh)$", last):
        return pre + last + "es"
    if re.search(r"[^aeiou]y$", last):
        return pre + last[:-1] + "ies"
    if last.endswith("fe"):
        return pre + last[:-2] + "ves"
    return pre + last + "s"


def count_phrase(label: str, n: int) -> str:
    if n == 1:
        return fix_articles(f"a {label}")
    return f"{NUM_WORDS[n] if n < len(NUM_WORDS) else n} {pluralize(label)}"


def join_list(items: list[str]) -> str:
    if len(items) <= 1:
        return "".join(items)
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def mentioned(label: str, text: str) -> bool:
    words = set(re.findall(r"[a-z]+", text.lower()))
    forms = SYNONYMS.get(label, set()) | {label, pluralize(label)}
    return any((f in words) if " " not in f else (f in text.lower()) for f in forms)


def _similar(a: str, b: str) -> float:
    A, B = set(a.lower().split()), set(b.lower().split())
    return len(A & B) / max(1, len(A | B))


# ---------- neural layer (optional) ----------
GRAMMAR_MODEL = os.getenv("GRAMMAR_MODEL", "vennify/t5-base-grammar-correction")
_neural = {"enabled": os.getenv("ENABLE_GRAMMAR_MODEL", "1") == "1"}


@lru_cache(maxsize=1)
def _grammar_pipe():
    from transformers import pipeline
    return pipeline("text2text-generation", model=GRAMMAR_MODEL)


def neural_fix(text: str) -> str:
    if not _neural["enabled"] or not text:
        return text
    try:
        out = _grammar_pipe()(f"grammar: {text}", max_new_tokens=64, num_beams=4)[0]["generated_text"].strip()
    except Exception as e:  # model unavailable: switch off for this process instead of retrying every call
        print(f"[info] grammar model disabled: {e}")
        _neural["enabled"] = False
        return text
    ratio = SequenceMatcher(None, text.lower(), out.lower()).ratio()
    if ratio < 0.8 or abs(len(out) - len(text)) > 0.25 * len(text) + 8:
        return text
    return out.rstrip(".")


def polish(raw: str) -> str:
    return clean_caption(neural_fix(clean_caption(raw)))


# ---------- final narration ----------
CROP_MIN_SCORE = float(os.getenv("CROP_MIN_SCORE", "-1.3"))  # length-normalised log-prob


def _keep_closeup(c, caption: str, kept: list[str]) -> str | None:
    text, label, score = (c, None, 0.0) if isinstance(c, str) else (c["caption"], c.get("label"), c.get("score", 0.0))
    p = polish(text)
    if not p or score < CROP_MIN_SCORE:
        return None                                   # low-confidence beam: usually word salad
    if label and not mentioned(label, p):
        return None                                   # caption is about something else than the crop
    if _similar(p, caption) >= 0.6 or any(_similar(p, q) >= 0.6 for q in kept):
        return None                                   # repeats what we've already said
    return p


def compose(caption_raw: str, labels: list[str], crop_captions: list | None = None,
            mode: str = "describe", max_closeups: int = 2) -> dict:
    """Display caption, object chips, polished close-ups and the spoken narration.

    crop_captions: strings, or dicts {"label", "caption", "score"} from Captioner.analyze.
    """
    caption = polish(caption_raw)
    counts = Counter(labels)
    closeups: list[str] = []
    for c in crop_captions or []:
        kept = _keep_closeup(c, caption, closeups)
        if kept:
            closeups.append(kept)

    # Flickr8k captions are sentences on their own ("a dog runs through the grass")
    parts = [sentence(caption)]
    told = " ".join([caption] + closeups[:max_closeups])
    extras = [count_phrase(lbl, n) for lbl, n in counts.most_common() if not mentioned(lbl, told)][:5]
    if extras:
        parts.append(sentence(f"I can also see {join_list(extras)}"))
    if mode != "live":
        parts += [sentence(f"Up close, {c}") for c in closeups[:max_closeups]]

    return {
        "caption": sentence(caption),
        "objects": [{"label": lbl, "count": n, "phrase": count_phrase(lbl, n)} for lbl, n in counts.most_common()],
        "closeups": [sentence(c) for c in closeups[:max_closeups]],
        "narration": " ".join(p for p in parts if p),
    }
