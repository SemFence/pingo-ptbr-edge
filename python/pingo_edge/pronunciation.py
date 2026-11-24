"""Respelling of English terms before the pt-BR phonemizer (which reads English words with Portuguese rules).

Applied to the input text only; the text you display keeps its normal spelling. A short list that is easy to extend: `(regex, how it is said)`.
The respelling uses Portuguese accents to guide the phonemizer (see docs/EXPERIMENTS.md, section 7).
"""
import re

PRON = [
    (r"\bHugging ?Face\b", "Ráguin Feice"), (r"\bGitHub\b", "Guit rub"), (r"\bstartups?\b", "istártape"),
    (r"\bdeploy\b", "diplói"), (r"\bMVP\b", "ême vê pê"), (r"\bAPI\b", "a pê i"), (r"\bGPU\b", "gê pê ú"),
    (r"\bpitch\b", "pítxi"), (r"\bfeedback\b", "fídbéque"), (r"\bdatasets?\b", "dêitasséti"),
    (r"\bvocoder\b", "vôcôder"), (r"\bdecoder\b", "decôder"), (r"\bencoder\b", "encôder"),
    (r"\bONNX\b", "ôniks"), (r"\bMLX\b", "ême éle xis"), (r"\bWebAssembly\b", "uébassémbli"),
    (r"\bbrowser\b", "bráuzer"), (r"\bcloud\b", "cláude"), (r"\bsoftware\b", "sófiuér"),
]


def spoken(text):
    for a, b in PRON:
        text = re.sub(a, b, text, flags=re.I if a.islower() else 0)
    return text


# The phonemizer puts a pause (<sp>) between the two words of "Guit rub"; GitHub is said as one word. The respelling keeps the
# space (so the phonemizer gives /x/ for the "r"), and this afterwards removes the pause and softens the vowel (the English
# /ʌ/ is closer to ɐ than to u): ɡ i t <sp> x u b -> ɡ i t x ɐ b.
JOIN = ("ɡ", "i", "t", "<sp>", "x", "u", "b")


def joined(ph):
    """Return `ph` (a tiny_voca Phonemized) without the pause inside "Guit rub"."""
    import dataclasses
    sym = list(ph.symbols); n = len(JOIN); i = 0; drop = []
    while i + n <= len(sym):
        if tuple(sym[i:i + n]) == JOIN: drop.append(i + 3); i += n
        else: i += 1
    if not drop: return ph
    from tiny_voca.text import SYMBOLS
    sym = [("ɐ" if (k - 2) in drop else x) for k, x in enumerate(sym)]; ids = [SYMBOLS.index("ɐ") if (k - 2) in drop else x for k, x in enumerate(ph.ids)]
    keep = [k for k in range(len(sym)) if k not in drop]
    return dataclasses.replace(ph, symbols=tuple(sym[k] for k in keep), ids=tuple(ids[k] for k in keep),
                               stress=tuple(ph.stress[k] for k in keep))
