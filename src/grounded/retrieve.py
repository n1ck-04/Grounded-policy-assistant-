"""Three ways to find the chunks that answer a question.

    bm25     Keyword search. Scores chunks by how many query words they contain,
             weighting rare words more (BM25, the classic search-engine formula).
             Great when the user uses the document's exact terms ("SCBMF",
             "cooling-off"); blind to synonyms ("fraud committee" vs "Special
             Committee of the Board").

    dense    Meaning-based search. Embeds the question and every chunk as vectors
             and ranks by cosine similarity. Handles paraphrases; can miss exact
             terms, numbers and acronyms.

    hybrid   Runs both and merges the rankings with Reciprocal Rank Fusion (RRF):
                 score(chunk) = sum over rankers of 1 / (RRF_K + rank)
             RRF uses only ranks, not raw scores, so we don't have to make BM25
             scores and cosine similarities comparable. RRF_K = 60 is the value
             from the original paper and dampens the effect of a single top rank.

All three return the same thing: a list of (chunk, score), best first.
"""

from __future__ import annotations

import re

import numpy as np
from rank_bm25 import BM25Okapi

from grounded import config
from grounded.chunking import Chunk

STOPWORDS = set("""a an and are as at be by for from has have in is it its of on or shall
that the their them they this to was were which will with what when where who how can
does do i my me should under any all""".split())


def tokenize(text: str) -> list[str]:
    """Lower-case words and numbers; keeps things like 'rs.5' -> 'rs', '5' and '3.3.4' as one token."""
    tokens = re.findall(r"\d+(?:\.\d+)*|[a-z]+", text.lower())
    return [t for t in tokens if t not in STOPWORDS]


class BM25Retriever:
    name = "bm25"

    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.index = BM25Okapi([tokenize(c.text) for c in chunks])

    def search(self, query: str, k: int = config.TOP_K) -> list[tuple[Chunk, float]]:
        scores = self.index.get_scores(tokenize(query))
        order = np.argsort(-scores)[:k]
        return [(self.chunks[i], float(scores[i])) for i in order]


class DenseRetriever:
    name = "dense"

    def __init__(self, chunks: list[Chunk], embedder):
        self.chunks, self.embedder = chunks, embedder
        self.matrix = embedder.embed_documents([c.text for c in chunks])

    def search(self, query: str, k: int = config.TOP_K) -> list[tuple[Chunk, float]]:
        scores = self.matrix @ self.embedder.embed_query(query)
        order = np.argsort(-scores)[:k]
        return [(self.chunks[i], float(scores[i])) for i in order]


def rrf_merge(rankings: list[list[Chunk]], k: int = config.RRF_K) -> list[tuple[Chunk, float]]:
    """Reciprocal Rank Fusion of several ranked lists."""
    scores: dict[str, float] = {}
    by_id: dict[str, Chunk] = {}
    for ranking in rankings:
        for rank, chunk in enumerate(ranking, start=1):
            scores[chunk.id] = scores.get(chunk.id, 0.0) + 1.0 / (k + rank)
            by_id[chunk.id] = chunk
    return sorted(((by_id[i], s) for i, s in scores.items()), key=lambda x: -x[1])


class HybridRetriever:
    name = "hybrid"

    def __init__(self, chunks: list[Chunk], embedder, depth: int = 20):
        self.bm25 = BM25Retriever(chunks)
        self.dense = DenseRetriever(chunks, embedder)
        self.depth = depth            # how many results each ranker contributes before fusion

    def search(self, query: str, k: int = config.TOP_K) -> list[tuple[Chunk, float]]:
        a = [c for c, _ in self.bm25.search(query, self.depth)]
        b = [c for c, _ in self.dense.search(query, self.depth)]
        return rrf_merge([a, b])[:k]


def build_retriever(kind: str, chunks: list[Chunk], embedder=None):
    if kind == "bm25":
        return BM25Retriever(chunks)
    if kind == "dense":
        return DenseRetriever(chunks, embedder)
    if kind == "hybrid":
        return HybridRetriever(chunks, embedder)
    raise ValueError(f"unknown retriever '{kind}'")
