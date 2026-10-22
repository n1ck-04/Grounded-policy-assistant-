"""Check an answer against the excerpts it claims to be based on.

The model must back every answer with citations of the form
    {"chunk_id": "customer_liability:8", "quote": "Within 3 working days | Zero liability"}
and the quote must be copied word for word from that chunk.

Checks
    1. Every cited chunk_id is one of the excerpts the model was actually shown.
    2. Every quote appears in that chunk (ignoring case, whitespace and quote
       styles; "..." may join two pieces from the same chunk).
    3. An answer that doesn't refuse must have at least one citation.
    4. A refusal must not carry citations (it said the excerpts don't answer).

Why this matters
    A model can write a fluent answer and attach a real-looking clause number
    to it. Requiring an exact quote, and verifying it, turns "trust me" into
    something a program can check. It doesn't prove the answer is correct
    (the quote might not support the claim), but it rules out invented sources.
"""

from __future__ import annotations

import re

from grounded.chunking import Chunk


def _norm(text: str) -> str:
    t = text.lower()
    t = t.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    t = t.replace("–", "-").replace("—", "-").replace("₹", "rs")
    t = re.sub(r"[\"']", "", t)
    return re.sub(r"\s+", " ", t).strip()


def quote_in_chunk(quote: str, chunk_text: str) -> bool:
    text = _norm(chunk_text)
    pieces = [p.strip() for p in re.split(r"\.\.\.|…", quote) if p.strip()]
    return bool(pieces) and all(_norm(p) in text for p in pieces)


def validate(result: dict, shown: list[Chunk]) -> list[str]:
    problems = []
    by_id = {c.id: c for c in shown}
    citations = result.get("citations") or []
    if not isinstance(result.get("answer"), str) or not result.get("answer", "").strip():
        problems.append("the answer text is empty")
    if result.get("refused"):
        if citations:
            problems.append("a refusal should not include citations")
        return problems
    if not citations:
        problems.append("cite at least one excerpt for a non-refusal answer")
    for i, c in enumerate(citations, 1):
        cid, quote = c.get("chunk_id"), c.get("quote", "")
        if cid not in by_id:
            problems.append(f"citation {i}: '{cid}' is not one of the excerpts you were given")
        elif not quote_in_chunk(quote, by_id[cid].text):
            problems.append(f"citation {i}: the quote is not an exact copy of text in {cid}")
    return problems
