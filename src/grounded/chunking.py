"""Split the parsed documents into chunks that the search indexes.

Two strategies, compared in the evaluation:

    fixed    windows of 200 words with a 40-word overlap, ignoring structure.
             The common default in RAG tutorials. Simple, but a window can start
             mid-sentence and mix two unrelated clauses.

    clause   one chunk per clause group, following the document's own numbering:
             "10", "10(i)", "10(ii)" -> one chunk for clause 10;
             "3.3.1" ... "3.3.4"      -> one chunk for clause 3.3.
             Groups longer than CLAUSE_MAX_WORDS are split at block boundaries.
             Every chunk starts with the document name and the section heading,
             so a chunk still makes sense on its own.

Every chunk records which clause ids it contains. That's what makes
citations checkable: the evaluation knows the correct clause for each test
question, and can ask "did the retrieved/cited chunk contain it?".
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from grounded import config

SHORT_NAMES = {
    "customer_liability": "RBI circular on limiting customer liability in unauthorised electronic banking transactions (2017)",
    "digital_lending": "RBI Digital Lending Directions (2025)",
    "fraud_risk": "RBI Master Directions on Fraud Risk Management (2024)",
}


@dataclass
class Chunk:
    id: str
    doc: str
    clauses: list[str]
    heading: str
    text: str
    meta: dict = field(default_factory=dict)

    def covers(self, gold: str) -> bool:
        """True if this chunk contains the gold clause (or one of its sub-clauses).
        gold looks like "customer_liability:8" or "fraud_risk:3.3.4"."""
        doc, _, clause = gold.partition(":")
        if doc != self.doc:
            return False
        return any(c == clause or c.startswith(clause + ".") or c.startswith(clause + "(") for c in self.clauses)


def load_blocks(path=config.SECTIONS_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def group_key(clause: str) -> str:
    """'10(ii)' -> '10';  '3.3.4' -> '3.3';  '2.1.1.2' -> '2.1';  '7' -> '7';  'Annex II' -> 'Annex II'."""
    base = clause.split("(")[0]
    parts = base.split(".")
    return ".".join(parts[:2]) if len(parts) > 2 else base


def _unique(seq):
    return list(dict.fromkeys(seq))


def clause_chunks(blocks: list[dict], max_words: int = config.CLAUSE_MAX_WORDS) -> list[Chunk]:
    chunks: list[Chunk] = []
    by_doc: dict[str, list[dict]] = {}
    for b in blocks:
        by_doc.setdefault(b["doc"], []).append(b)
    for doc, bs in by_doc.items():
        groups: list[tuple[str, list[dict]]] = []
        for b in bs:
            key = group_key(b["clause"])
            if groups and groups[-1][0] == key:
                groups[-1][1].append(b)
            else:
                groups.append((key, [b]))
        # A group that is only headings ("CHAPTER III", "3. Early Detection of Frauds")
        # is merged into the group that follows, so no chunk is just a title.
        merged: list[tuple[str, list[dict]]] = []
        pending: list[dict] = []
        for key, members in groups:
            if all(b["kind"] == "heading" for b in members):
                pending += members
                continue
            merged.append((key, pending + members))
            pending = []
        if pending and merged:
            merged[-1][1].extend(pending)
        for key, members in merged:
            parts, current, words = [], [], 0
            for b in members:
                n = len(b["text"].split())
                if current and words + n > max_words:
                    parts.append(current)
                    current, words = [], 0
                current.append(b)
                words += n
            if current:
                parts.append(current)
            for i, part in enumerate(parts):
                heads = [b["text"] for b in part if b["kind"] == "heading"]
                numbered = [h for h in heads if h[:1].isdigit()]
                heading = (numbered or heads or [part[0]["heading"]])[-1]   # most specific heading
                header = f"{SHORT_NAMES.get(doc, doc)} | {heading}"
                body = "\n".join(b["text"] for b in part)
                cid = f"{doc}:{key}" + (f"#{i + 1}" if len(parts) > 1 else "")
                chunks.append(Chunk(cid, doc, _unique(b["clause"] for b in part), heading, f"{header}\n{body}"))
    return chunks


def fixed_chunks(blocks: list[dict], size: int = config.FIXED_CHUNK_WORDS,
                 overlap: int = config.FIXED_OVERLAP_WORDS) -> list[Chunk]:
    chunks: list[Chunk] = []
    by_doc: dict[str, list[dict]] = {}
    for b in blocks:
        by_doc.setdefault(b["doc"], []).append(b)
    for doc, bs in by_doc.items():
        words, owner = [], []                       # each word remembers which clause it came from
        for b in bs:
            w = b["text"].split()
            words += w
            owner += [b["clause"]] * len(w)
        step = size - overlap
        for n, start in enumerate(range(0, max(1, len(words) - overlap), step)):
            end = min(start + size, len(words))
            text = " ".join(words[start:end])
            chunks.append(Chunk(f"{doc}#w{n + 1}", doc, _unique(owner[start:end]), "",
                                f"{SHORT_NAMES.get(doc, doc)}\n{text}"))
            if end == len(words):
                break
    return chunks


STRATEGIES = {"clause": clause_chunks, "fixed": fixed_chunks}


def build_chunks(strategy: str = "clause", blocks: list[dict] | None = None) -> list[Chunk]:
    return STRATEGIES[strategy](blocks if blocks is not None else load_blocks())
