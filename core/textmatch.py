"""Lightweight, dependency-free text matching used to answer *specific* questions.

* stemming            "reviewed" / "reviews" / "review"  ->  "review"
* abbreviation handling  "CTO" <-> "Chief Technology Officer"
* IDF weighting       rare words (names, numbers, product terms) matter more
* "units"             one unit per meaningful question word; a unit is matched by the
                      word's stem, a synonym, or its full abbreviation expansion
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, Iterable, List, Sequence, Set

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Words that make a question a question but carry no searchable meaning.
FILLERS = {
    "what", "who", "whom", "whose", "which", "when", "where", "why", "how", "does", "did", "do",
    "is", "are", "was", "were", "can", "could", "should", "would", "will", "shall", "may", "might",
    "tell", "explain", "describe", "give", "list", "state", "mention", "mentioned", "please",
    "many", "much", "often", "long", "far", "soon", "quickly", "fast", "way", "kind", "type",
    "need", "needed", "get", "got", "use", "used", "document", "documents", "text", "pdf",
    "file", "according", "say", "says", "said", "about", "regarding", "thing", "things",
    "know", "mean", "means", "meant", "exactly", "really", "also", "like", "etc",
    "page", "pages", "company",
    "types", "kinds", "ways", "limit", "limits", "maximum", "minimum", "max", "min", "amount", "total", "number", "value",
    "rate", "level", "size", "length", "located", "location", "locate", "situated", "based",
    "happen", "happens", "work", "works", "happened", "mentioned", "include", "includes", "included",
}
STOP = set(ENGLISH_STOP_WORDS) | FILLERS

ABBREVIATIONS: Dict[str, str] = {
    "ceo": "chief executive officer", "cto": "chief technology officer",
    "cfo": "chief financial officer", "coo": "chief operating officer",
    "cio": "chief information officer", "ciso": "chief information security officer",
    "hr": "human resources", "pto": "paid time off", "wfh": "work from home",
    "ml": "machine learning", "ai": "artificial intelligence",
    "nlp": "natural language processing", "rag": "retrieval augmented generation",
    "llm": "large language model", "ui": "user interface", "api": "application programming interface",
    "ta": "teaching assistant", "faq": "frequently asked questions", "kpi": "key performance indicator",
    "roi": "return on investment", "sla": "service level agreement",
}
SYNONYMS: Dict[str, Set[str]] = {
    "cost": {"price", "priced", "fee", "charge", "pricing"},
    "price": {"cost", "fee", "priced", "pricing"},
    "begin": {"start"}, "start": {"begin"}, "end": {"finish", "terminate"},
    "buy": {"purchase"}, "purchase": {"buy"},
    "boss": {"manager", "supervisor"}, "manager": {"boss", "supervisor"},
    "employee": {"staff", "worker"}, "staff": {"employee"},
    "stolen": {"lost", "theft"}, "salary": set(),
    "class": {"lecture", "session"}, "classes": {"lecture", "lectures", "session"},
    "lecture": {"class", "session"}, "professor": {"instructor", "lecturer", "teacher"},
    "teacher": {"professor", "instructor", "lecturer"}, "instructor": {"professor", "teacher"},
    "penalty": {"lose", "deduct", "deduction", "fine", "penalise", "penalize", "reduce"},
    "hand": {"submit", "submission", "turn"}, "handing": {"submit", "submission"},
    "approve": {"approval", "permit", "granted"}, "recommend": {"suggested", "advise"},
    "worth": {"contribute", "contributes", "weight", "percent"},
    "enrol": {"enroll", "register", "pass", "passed", "prerequisite"},
    "enroll": {"enrol", "register", "pass", "passed", "prerequisite"},
    "refuse": {"reject", "accept", "accepted"}, "refused": {"reject", "rejected", "accepted"},
    "consequence": {"result", "results", "reported", "punish", "penalty"},
    "meet": {"office", "hours", "contact", "available"}, "contact": {"email", "reach"},
    "term": {"semester", "weeks"}, "semester": {"term"},
    "kept": {"stored", "retained", "saved", "stored"}, "keep": {"store", "retain", "save"},
}
NUMBER_WORDS = {"one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
                "eleven", "twelve", "twenty", "thirty", "forty", "fifty", "hundred", "thousand",
                "million", "half", "twice", "once", "single", "first", "second", "third"}

IRREGULAR = {"taught": "teach", "held": "hold", "paid": "pay", "kept": "keep", "bought": "buy",
             "wrote": "write", "written": "write", "gave": "give", "given": "give", "ran": "run",
             "built": "build", "sold": "sell", "led": "lead", "made": "make", "began": "begin",
             "begun": "begin", "chose": "choose", "chosen": "choose", "left": "leave", "met": "meet",
             "won": "win", "known": "know", "thought": "think", "spent": "spend", "sent": "send",
             "lost": "lose", "found": "find", "took": "take", "taken": "take", "got": "get",
             "went": "go", "came": "come", "saw": "see", "seen": "see", "told": "tell",
             "brought": "bring", "stood": "stand", "understood": "understand", "became": "become"}

_WORD = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)*")


def stem(word: str) -> str:
    w = word.lower().strip("'")
    if w in IRREGULAR:
        w = IRREGULAR[w]
    if w.endswith("'s"):
        w = w[:-2]
    if len(w) <= 3 or w.isdigit() or any(c.isdigit() for c in w):
        return w
    if w.endswith("ies") and len(w) > 4:
        w = w[:-3] + "y"
    elif w.endswith("ied") and len(w) > 4:
        w = w[:-3] + "y"
    elif w.endswith("ing") and len(w) > 5:
        w = w[:-3]
    elif w.endswith("ed") and len(w) > 4:
        w = w[:-2]
    elif w.endswith("ment") and len(w) > 7:
        w = w[:-4]
    elif w.endswith("ly") and len(w) > 5:
        w = w[:-2]
    elif w.endswith("es") and len(w) > 4 and w[-3] in "sxz":
        w = w[:-2]
    elif w.endswith("s") and not w.endswith(("ss", "us", "is")) and len(w) > 3:
        w = w[:-1]
    if len(w) > 3 and w.endswith("e"):
        w = w[:-1]
    if len(w) > 3 and w[-1] == w[-2] and w[-1] not in "sl":   # "planned" -> "plann" -> "plan"
        w = w[:-1]
    return w


def words(text: str) -> List[str]:
    return _WORD.findall(text.lower().replace("’", "'").replace("'s ", " ").replace("'s", ""))


def stems(text: str) -> Set[str]:
    return {stem(w) for w in words(text)}


def content_words(text: str) -> List[str]:
    return [w for w in words(text) if w not in STOP and (len(w) > 1 or w.isdigit())]


def build_units(question: str) -> List[List[Set[str]]]:
    """One unit per content word; each unit is a list of alternatives, each alternative a
    set of stems that must ALL be present for the unit to count as matched."""
    units: List[List[Set[str]]] = []
    seen: Set[str] = set()
    for w in content_words(question):
        if w in NUMBER_WORDS:          # "three types of ..." -> quantifier, not a searchable term
            continue
        s = stem(w)
        if s in seen:
            continue
        seen.add(s)
        alts: List[Set[str]] = [{s}]
        if w in ABBREVIATIONS:
            alts.append({stem(x) for x in ABBREVIATIONS[w].split()})
        for syn in SYNONYMS.get(w, ()):
            alts.append({stem(syn)})
        units.append(alts)
    return units


def _has(token: str, present: Set[str]) -> bool:
    """Exact stem match, or one stem is a prefix of the other ("complain" ~ "complaint")."""
    if token in present:
        return True
    if len(token) < 4:
        return False
    return any((len(p) >= 5 and len(token) >= 5 and (p.startswith(token) or token.startswith(p)))
               or (len(token) >= 4 and len(p) > len(token) + 2 and p.endswith(token))   # compound: textbook ~ book
               for p in present)


def unit_matched(unit: List[Set[str]], present: Set[str]) -> bool:
    return any(all(_has(t, present) for t in alt) for alt in unit)


class CorpusStats:
    """Document frequencies over stemmed chunk text (chunk = document unit)."""

    def __init__(self, texts: Iterable[str]) -> None:
        texts = list(texts)
        self.n = max(len(texts), 1)
        self.df: Counter = Counter()
        for t in texts:
            self.df.update(stems(t))

    def idf(self, token: str) -> float:
        return math.log(1.0 + self.n / (1.0 + self.df.get(token, 0)))

    def unit_weight(self, unit: List[Set[str]], power: float = 1.0) -> float:
        # a unit is only as "rare" as its most common spelling, an unseen word is the rarest
        alt = unit[0]
        return max(self.idf(t) for t in alt) ** power


def coverage(units: Sequence[List[Set[str]]], present: Set[str], stats: CorpusStats,
             power: float = 1.0) -> float:
    """IDF-weighted fraction of question units found in `present` (0..1)."""
    if not units:
        return 0.0
    total = sum(stats.unit_weight(u, power) for u in units)
    got = sum(stats.unit_weight(u, power) for u in units if unit_matched(u, present))
    return got / total if total else 0.0


def get_stats(store) -> CorpusStats:
    """Cache corpus statistics on the vector store object."""
    st = getattr(store, "_corpus_stats", None)
    if st is None or st.n != max(len(store.chunks), 1):
        st = CorpusStats(c.text for c in store.chunks)
        store._corpus_stats = st
    return st


def unseen_fraction(units: Sequence[List[Set[str]]], stats: CorpusStats) -> float:
    """Share of the question's terms that occur NOWHERE in the documents (exact, prefix or
    compound match). A high value means the question asks about something the text never mentions."""
    if not units:
        return 0.0
    vocab = set(stats.df)
    unseen = sum(1 for u in units if not any(all(_has(t, vocab) for t in alt) for alt in u))
    return unseen / len(units)
