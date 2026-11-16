"""Turns raw model output into clean, grammatical spoken English.

Two layers:
1. Deterministic rules (always on): removes BLIP artifacts ("arafed"), echoed prompts,
   stutters and repeated phrases, dangling words, wrong a/an, spacing, capitals, punctuation;
   pluralises and counts objects; joins lists naturally.
2. Optional neural grammar correction (T5) applied only to the caption, accepted only when
   the change is small, so the model can't hallucinate new content.
"""
from __future__ import annotations

import os
import re
from collections import Counter
from difflib import SequenceMatcher
from functools import lru_cache

PROMPT_ECHOES = [
    "a detailed picture of", "a picture of", "a photo of", "a photograph of", "an image of",
    "a close up of", "a close - up of", "there is", "there are",
]
ARTIFACT_RE = re.compile(r"\b(ara+f+[a-z]*|arafy|araffe|arafed)\b", re.I)
DANGLING = {"a", "an", "the", "with", "and", "of", "in", "on", "at", "to", "for", "by",
            "next", "near", "from", "while", "that", "is", "are", "its", "his", "her", "their"}
AN_EXCEPTIONS = ("uni", "use", "usu", "uti", "eu", "one", "once", "ubiq", "ura")
A_EXCEPTIONS = ("hour", "honest", "honou", "heir")
NUM_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
IRREGULAR = {"person": "people", "man": "men", "woman": "women", "child": "children",
             "mouse": "mice", "foot": "feet", "tooth": "teeth", "sheep": "sheep", "knife": "knives",
             "wine glass": "wine glasses", "skis": "skis", "scissors": "scissors", "bus": "buses"}
SYNONYMS = {
    "person": {"person", "people", "man", "men", "woman", "women", "boy", "girl", "child",
               "children", "kid", "kids", "lady", "guy", "player", "persons", "baby"},
    "dining table": {"table", "dining table"}, "cell phone": {"phone", "cellphone", "cell phone"},
    "couch": {"couch", "sofa"}, "tv": {"tv", "television"}, "motorcycle": {"motorcycle", "motorbike", "bike"},
    "bicycle": {"bicycle", "bike"}, "car": {"car", "cars", "vehicle"},
}


# ---------- rule layer ----------
def _collapse_repeats(words: list[str]) -> list[str]:
    out: list[str] = []
    for w in words:
        if out and w.lower() == out[-1].lower():
            continue
        out.append(w)
    changed = True
    while changed:  # remove repeated n-grams: "on a bench on a bench" -> "on a bench"
        changed = False
        for n in range(6, 1, -1):
            for i in range(len(out) - 2 * n + 1):
                a = [w.lower() for w in out[i:i + n]]
                b = [w.lower() for w in out[i + n:i + 2 * n]]
                if a == b:
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
        if re.match(r"^[aefhilmnorsx]$", lw):  # single letters spoken with vowel sound: "an X"
            vowel = True
        new = "an" if vowel else "a"
        if art[0].isupper():
            new = new.capitalize()
        return f"{new} {word}"
    return re.sub(r"\b(a|an|A|An)\s+([A-Za-z][\w'-]*)", repl, text)


def clean_caption(raw: str) -> str:
    """Model caption -> tidy noun phrase / sentence, no final period."""
    t = (raw or "").strip()
    t = ARTIFACT_RE.sub("", t)
    low = t.lower()
    for p in PROMPT_ECHOES:
        if low.startswith(p + " "):
            t = t[len(p):].strip()
            break
    t = re.sub(r"\s+([,.!?;:])", r"\1", t)
    t = re.sub(r"[,;:]+\s*$", "", t)
    words = _collapse_repeats(t.replace(" ,", ",").split())
    while words and words[-1].lower().strip(",.") in DANGLING:
        words.pop()
    while words and words[0].lower() in {"and", "with", "of"}:
        words.pop(0)
    t = " ".join(words).strip(" ,.")
    t = re.sub(r"\s{2,}", " ", t)
    t = add_leading_article(t)
    t = fix_articles(t)
    return t or "an image I couldn't make out"


NO_ARTICLE = {"a", "an", "the", "this", "that", "these", "those", "some", "many", "several", "few",
              "his", "her", "their", "its", "my", "our", "your", "there", "it", "he", "she", "they", "i",
              "we", "people", "men", "women", "children", "kids", "two", "three", "four", "five", "six",
              "seven", "eight", "nine", "ten", "one", "lots", "all", "both", "each", "every", "no"}


def add_leading_article(t: str) -> str:
    """'brown dog sitting on a bench' -> 'a brown dog sitting on a bench' (BLIP often drops it)."""
    words = t.split()
    if not words:
        return t
    first = words[0].lower()
    if first in NO_ARTICLE or first.isdigit() or words[0][0].isupper() and len(words[0]) > 1 and words[0].isupper():
        return t
    plural_subject = any(w.lower() in {"are", "were"} for w in words[:4]) or (
        len(words) > 1 and first.endswith("s") and not first.endswith("ss") and words[1].lower().endswith("ing"))
    return t if plural_subject else "a " + t


def sentence(text: str) -> str:
    t = re.sub(r"\s{2,}", " ", text.strip())
    if not t:
        return ""
    t = t[0].upper() + t[1:]
    if t[-1] not in ".!?”\"":
        t += "."
    return t


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
    num = NUM_WORDS[n] if n < len(NUM_WORDS) else str(n)
    return f"{num} {pluralize(label)}"


def join_list(items: list[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def mentioned(label: str, caption: str) -> bool:
    words = set(re.findall(r"[a-z]+", caption.lower()))
    cap = caption.lower()
    forms = SYNONYMS.get(label, {label}) | {label, pluralize(label)}
    return any((f in words) if " " not in f else (f in cap) for f in forms)


def answer_sentence(question: str, answer: str) -> str:
    a = (answer or "").strip().rstrip(".")
    if not a:
        return ""
    la = a.lower()
    if la in {"yes", "no"}:
        return f"{la.capitalize()}."
    if la.isdigit() or la in NUM_WORDS:
        n = int(la) if la.isdigit() else NUM_WORDS.index(la)
        m = re.search(r"how many ([a-z ]+?)(?: are| is| do| can|\?|$)", question.lower())
        if m:
            noun = m.group(1).strip()
            num = NUM_WORDS[n] if n < len(NUM_WORDS) else str(n)
            if n == 1:
                noun = re.sub(r"(people)$", "person", noun)
                noun = noun[:-1] if noun.endswith("s") and not noun.endswith("ss") else noun
                return sentence(f"there is one {noun}")
            return sentence(f"there are {num} {noun}")
        return sentence(f"I count {a}")
    return sentence(f"it looks like {fix_articles(a)}" if len(a.split()) <= 3 else a)


# ---------- neural layer (optional) ----------
GRAMMAR_MODEL = os.getenv("GRAMMAR_MODEL", "vennify/t5-base-grammar-correction")
USE_GRAMMAR_MODEL = os.getenv("ENABLE_GRAMMAR_MODEL", "1") == "1"


@lru_cache(maxsize=1)
def _grammar_pipe():
    from transformers import pipeline
    return pipeline("text2text-generation", model=GRAMMAR_MODEL)


def neural_fix(text: str) -> str:
    if not USE_GRAMMAR_MODEL or not text:
        return text
    try:
        out = _grammar_pipe()(f"grammar: {text}", max_new_tokens=64, num_beams=4)[0]["generated_text"].strip()
    except Exception:
        return text
    # Safety: accept only small edits (no new content, nothing dropped).
    ratio = SequenceMatcher(None, text.lower(), out.lower()).ratio()
    if ratio < 0.8 or abs(len(out) - len(text)) > 0.25 * len(text) + 8:
        return text
    return out.rstrip(".")


# ---------- final narration ----------
def compose(caption_raw: str, labels: list[str], ocr_lines: list[str], question: str = "",
            answer: str = "", mode: str = "describe") -> dict:
    """Builds display caption and the full spoken narration."""
    caption = clean_caption(caption_raw)
    caption = clean_caption(neural_fix(caption))
    display = sentence(caption)

    counts = Counter(labels)
    extras = [count_phrase(l, n) for l, n in counts.most_common() if not mentioned(l, caption)][:5]
    text = " ".join(dict.fromkeys(s.strip() for s in ocr_lines if s.strip()))[:200]

    parts: list[str] = []
    if mode == "read":
        parts.append(f"The text reads: “{text}”." if text else "I couldn't find any readable text in this image.")
        parts.append(sentence(f"it appears on {caption}") if text else "")
    else:
        starts_sentence = re.match(r"^(there|it|this|he|she|they|i|we)\b", caption, re.I)
        parts.append(sentence(caption) if starts_sentence else sentence(f"I can see {caption}"))
        if extras:
            parts.append(sentence(f"I can also see {join_list(extras)}"))
        if text and mode != "live":
            parts.append(f"There is text that reads “{text}”.")
    if question and answer:
        parts.append(answer_sentence(question, answer))

    narration = " ".join(p for p in parts if p)
    return {
        "caption": display,
        "objects": [{"label": l, "count": n, "phrase": count_phrase(l, n)} for l, n in counts.most_common()],
        "text": text,
        "answer": answer_sentence(question, answer) if answer else None,
        "narration": narration,
    }
