import re

STOP = set("""a an and are as at be but by for from has have i if in is it its my of on or so that the
their them there they this to was were with you your we our me not no do does did would could can
just very really also than then too when what which who how about into out up one get got
good great bad nice best people say tell think""".split())
_tok = re.compile(r"[a-z0-9']+")


def stem(w: str) -> str:
    for suf in ("ingly", "edly", "ing", "ed", "ly", "es", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 3 and not w.endswith("ss"):
            return w[: -len(suf)]
    return w


def tokenize(text: str) -> list[str]:
    return [stem(w) for w in _tok.findall(text.lower().replace("'", "")) if w not in STOP and len(w) > 1]


def sentences(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"(?<=[.!?])\s+", text.strip()) if p.strip()]
