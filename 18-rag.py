"""Retrieval + answer composition for Ask JNTUH. Pure stdlib - no heavy deps."""
import json, math, os, re, urllib.request
from pathlib import Path

STOP = set("""a an the and or of to in for on is are was were be been being do does did doing
i me my we our you your he she it they them this that these those what which who whom how when where why
can could should would will shall may might must not no yes as at by from with about into over after before
""".split())

SYN = {"detention": "detain", "detained": "detain", "detains": "detain",
       "promotion": "promote", "promoted": "promote", "promote": "promote",
       "failed": "fail", "fails": "fail", "failing": "fail",
       "passing": "pass", "passes": "pass", "passed": "pass",
       "attendance": "attend", "attending": "attend", "attend": "attend"}

def _stem(t):
    if t in SYN:
        return SYN[t]
    for suf in ("ing", "edly", "ed", "es", "s"):
        if len(t) > len(suf) + 3 and t.endswith(suf):
            return t[: -len(suf)]
    return t

def tokenize(s):
    return [_stem(t) for t in re.findall(r"[a-z0-9]+", s.lower()) if t not in STOP and len(t) > 1]

class Retriever:
    def __init__(self, chunks_path):
        self.chunks = json.loads(Path(chunks_path).read_text())
        self.df, self.tf, self.dl = {}, [], []
        self.avgdl = 0.0
        for c in self.chunks:
            toks = tokenize(c["text"])
            self.dl.append(len(toks))
            tf = {}
            for t in toks:
                tf[t] = tf.get(t, 0) + 1
            self.tf.append(tf)
            for t in tf:
                self.df[t] = self.df.get(t, 0) + 1
        self.avgdl = sum(self.dl) / max(len(self.dl), 1)
        self.N = len(self.chunks)

    def search(self, query, k=6, k1=1.5, b=0.75):
        q = tokenize(query)
        scores = []
        for i, tf in enumerate(self.tf):
            s = 0.0
            for t in q:
                if t not in tf:
                    continue
                idf = math.log(1 + (self.N - self.df[t] + 0.5) / (self.df[t] + 0.5))
                s += idf * tf[t] * (k1 + 1) / (tf[t] + k1 * (1 - b + b * self.dl[i] / self.avgdl))
            if s > 0:
                scores.append((s, i))
        scores.sort(reverse=True)
        return [(self.chunks[i], round(s, 2)) for s, i in scores[:k]]

JUNK = re.compile(r"(?i)(procs?\.?\s*no|registrar|orders\s*[-–]\s*issued|kukatpally|proceedings of the|^sub\s*:|establish(ed)? by (govt|state)|e\s*-?\s*mail|phone\s*:|web\s*:|www\.|proposal to relax)")

def _clean(s):
    if JUNK.search(s):
        return False
    toks = s.split()
    if len(s) < 40 or len(toks) < 6:
        return False
    alpha = sum(c.isalpha() or c.isspace() for c in s) / len(s)
    if alpha < 0.75:
        return False
    single = sum(1 for t in toks if len(t.strip(".,;:()")) <= 1) / len(toks)
    if single > 0.25:
        return False
    norm = [t.strip(".,;:()").lower() for t in toks]
    return len(set(norm)) / len(norm) >= 0.6

def _sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z(0-9])", text) if _clean(s.strip())]

def extractive_answer(query, hits, retriever=None):
    """Compose a cited answer from the best sentences of the top chunks.
    Sentences are ranked by IDF-weighted coverage of query terms, so the
    distinctive terms (e.g. 'attendance') outweigh common ones ('required')."""
    import math
    q = set(tokenize(query))
    if retriever is not None:
        N = retriever.N
        w = {t: math.log(1 + N / (1 + retriever.df.get(t, 0))) for t in q}
    else:
        w = {t: 1.0 for t in q}
    total_w = sum(w.values()) or 1.0
    scored = []
    for rank, (chunk, cscore) in enumerate(hits[:8]):
        for s in _sentences(chunk["text"]):
            st = set(tokenize(s))
            matched = q & st
            if not matched:
                continue
            wcov = sum(w[t] for t in matched) / total_w
            scored.append((wcov * 10 - rank * 0.4 + min(len(st), 30) / 30, s, chunk))
    scored.sort(key=lambda x: -x[0])
    picked, seen = [], set()
    for _, s, c in scored:
        key = s[:60].lower()
        if key in seen:
            continue
        seen.add(key)
        picked.append({"sentence": s, "doc": c["doc"], "page": c["page"], "url": c["url"]})
        if len(picked) == 4:
            break
    return picked

def groq_answer(query, hits):
    """Optional generative mode when GROQ_API_KEY is set (free tier, llama-3.1-8b-instant)."""
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        return None
    ctx = "\n\n".join(f"[{h['doc']}, p.{h['page']}]\n{h['text'][:900]}" for h, _ in hits[:5])
    body = json.dumps({
        "model": "llama-3.1-8b-instant",
        "messages": [
            {"role": "system", "content": "You answer questions about JNTUH academic regulations using ONLY the provided excerpts. Be concise (under 150 words). Cite the source document and page like (R22 Regulations, p.5) after each claim. If the excerpts do not answer the question, say so."},
            {"role": "user", "content": f"Excerpts:\n{ctx}\n\nQuestion: {query}"},
        ],
        "temperature": 0.2,
        "max_tokens": 400,
    }).encode()
    req = urllib.request.Request("https://api.groq.com/openai/v1/chat/completions",
        data=body, headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return json.loads(r.read())["choices"][0]["message"]["content"].strip()
    except Exception:
        return None
