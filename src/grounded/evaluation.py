"""Evaluation harness: retrieval first (free, deterministic), then answers.

Retrieval metrics (answerable questions only; unanswerable ones have no gold)
    recall@k   share of a question's gold clauses found in the top-k chunks,
               averaged over questions. The model can only cite what it's given,
               so this caps everything downstream.
    hit@1      the top chunk contains a gold clause.
    mrr        mean reciprocal rank of the first chunk containing a gold clause
               (1.0 = always first, 0.5 = typically second, ...).

Answer metrics are computed in answer_eval.py (needs an API key): judged accuracy,
fact-check pass rate, correct refusals, citation hit rate, retries and cost.

Run:  python -m grounded.evaluation           (retrieval ablation; free)
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass

import matplotlib
import yaml

from grounded import config
from grounded.chunking import Chunk, build_chunks
from grounded.retrieve import build_retriever

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


@dataclass
class Question:
    id: str
    question: str
    type: str
    kind: str
    gold: list[str]
    facts: list[list[str]]


def load_testset(path=config.TESTSET_PATH) -> list[Question]:
    data = yaml.safe_load(open(path, encoding="utf-8"))
    return [Question(**q) for q in data["questions"]]


# ----------------------------------------------------------------------- text matching
def normalise(text: str) -> str:
    """Make fact matching robust to formatting: case, quotes, dashes, 5,000 vs 5000, % vs per cent."""
    t = text.lower()
    t = t.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    t = t.replace("–", "-").replace("—", "-").replace("₹", "rs ")
    t = re.sub(r"(?<=\d),(?=\d)", "", t)                 # 5,000 -> 5000
    t = re.sub(r"\s*%", " per cent", t)
    t = t.replace("percent", "per cent")
    return re.sub(r"\s+", " ", t)


def facts_present(answer: str, facts: list[list[str]]) -> bool:
    a = normalise(answer)
    return all(any(normalise(alt) in a for alt in fact) for fact in facts)


# --------------------------------------------------------------------------- retrieval
def covered(chunks: list[Chunk], gold: list[str]) -> list[str]:
    return [g for g in gold if any(c.covers(g) for c in chunks)]


def retrieval_scores(retriever, questions: list[Question], k: int = config.TOP_K) -> dict:
    rows = []
    for q in questions:
        if q.type != "answerable":
            continue
        ranked = [c for c, _ in retriever.search(q.question, k)]
        first = next((i for i, c in enumerate(ranked, 1) if any(c.covers(g) for g in q.gold)), None)
        rows.append({
            "id": q.id, "kind": q.kind,
            "recall": len(covered(ranked, q.gold)) / len(q.gold),
            "hit1": float(first == 1),
            "rr": 1.0 / first if first else 0.0,
            "retrieved": [c.id for c in ranked],
            "context_words": sum(len(c.text.split()) for c in ranked),
        })
    n = len(rows)
    by_kind = {}
    for kind in sorted({r["kind"] for r in rows}):
        rs = [r for r in rows if r["kind"] == kind]
        by_kind[kind] = round(sum(r["recall"] for r in rs) / len(rs), 3)
    return {
        f"recall@{k}": round(sum(r["recall"] for r in rows) / n, 3),
        "hit@1": round(sum(r["hit1"] for r in rows) / n, 3),
        "mrr": round(sum(r["rr"] for r in rows) / n, 3),
        "recall_by_kind": by_kind,
        "avg_context_words": round(sum(r["context_words"] for r in rows) / n),
        "misses": [r["id"] for r in rows if r["recall"] < 1],
        "rows": rows,
    }


def plot_ablation(results: dict, path):
    strategies, retrievers = ["fixed", "clause"], ["bm25", "dense", "hybrid"]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    width = 0.35
    for i, s in enumerate(strategies):
        vals = [results[f"{s}/{r}"][f"recall@{config.TOP_K}"] for r in retrievers]
        bars = ax.bar([x + (i - 0.5) * width for x in range(len(retrievers))], vals, width,
                      label=f"{s} chunks")
        ax.bar_label(bars, fmt="%.2f", fontsize=8)
    ax.set_xticks(range(len(retrievers)), ["BM25 (keywords)", "Dense (meaning)", "Hybrid (RRF)"])
    ax.set(ylim=(0, 1.1), ylabel=f"Recall@{config.TOP_K}", title="Did the top 5 chunks contain the right clause?")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    from grounded.embed import CachedEmbedder, Embedder

    questions = load_testset()
    embedder = CachedEmbedder(Embedder())
    results = {}
    for strategy in ["fixed", "clause"]:
        chunks = build_chunks(strategy)
        for kind in ["bm25", "dense", "hybrid"]:
            t = time.time()
            r = build_retriever(kind, chunks, embedder)
            scores = retrieval_scores(r, questions)
            scores["chunks"] = len(chunks)
            scores["ms_per_query"] = round(1000 * (time.time() - t) / len(questions), 1)
            results[f"{strategy}/{kind}"] = scores
            print(f"{strategy:<7}{kind:<7} recall@{config.TOP_K} {scores[f'recall@{config.TOP_K}']:.3f}  "
                  f"hit@1 {scores['hit@1']:.3f}  MRR {scores['mrr']:.3f}  ctx {scores['avg_context_words']}w  "
                  f"misses {scores['misses']}")
    config.REPORTS_DIR.mkdir(exist_ok=True)
    (config.REPORTS_DIR / "retrieval.json").write_text(json.dumps(results, indent=2))
    plot_ablation(results, config.REPORTS_DIR / "retrieval.png")


if __name__ == "__main__":
    main()
