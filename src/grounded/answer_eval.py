"""End-to-end evaluation: run the assistant on all 65 test questions and score the answers.

Three systems are compared, all using the same test set:
    naive           fixed chunks + dense search + plain prompt (no citations, no refusal rule, no checks)
    grounded-fixed  fixed chunks + dense search  + grounded prompt + verified quotes + refusals
    grounded-clause clause chunks + hybrid search + grounded prompt + verified quotes + refusals

How answers are scored
    Two independent signals, because each has blind spots:

    1. Fact check (deterministic): does the answer contain every required fact from
       testset.yaml (e.g. "21 days", "5,000")? Cheap and reproducible, but a correct
       answer phrased differently can fail, and a vague fact like "not" can pass by luck.

    2. LLM judge: a second Claude model sees the question, the TEXT OF THE GOLD
       CLAUSES and the answer, and labels it correct / partial / incorrect (answerable
       questions) or abstained / answered (unanswerable ones). It catches paraphrases
       the fact check misses. The judge never sees which system produced the answer.

    Reported "accuracy" = judged correct. We also report how often the two signals
    agree, so the reader can see how much to trust the judge.

Run:  python -m grounded.answer_eval --systems naive grounded-fixed grounded-clause
"""

from __future__ import annotations

import argparse
import json
import time

from grounded import config
from grounded.answer import Answerer, ClaudeClient
from grounded.chunking import build_chunks, load_blocks
from grounded.evaluation import facts_present, load_testset
from grounded.retrieve import build_retriever

SYSTEMS = {
    "naive": ("fixed", "dense", "naive"),
    "grounded-fixed": ("fixed", "dense", "grounded"),
    "grounded-clause": ("clause", "hybrid", "grounded"),
}

JUDGE_MODEL = "claude-haiku-5-5"
JUDGE_SYSTEM = """You grade answers from an assistant that answers questions about RBI regulations.
You are given the question, the reference text from the regulation (or a note that the documents don't
cover the question), and the assistant's answer. Grade only what the answer says, not its style.

If reference text is given:
  correct    the answer states the key requirement(s) accurately, with the right numbers/conditions
  partial    the answer is on the right track but misses or blurs an important condition or number
  incorrect  the answer is wrong, contradicts the reference, or refuses to answer
If the documents don't cover the question:
  abstained  the answer declines, or clearly says the documents don't specify this
  answered   the answer gives a substantive answer anyway (even if hedged)

Respond by calling grade."""

GRADE_TOOL = {
    "name": "grade",
    "description": "Record the grade.",
    "input_schema": {
        "type": "object",
        "properties": {
            "label": {"type": "string", "enum": ["correct", "partial", "incorrect", "abstained", "answered"]},
            "reason": {"type": "string", "description": "One sentence"},
        },
        "required": ["label", "reason"],
    },
}


def gold_text(gold: list[str], blocks: list[dict]) -> str:
    out = []
    for g in gold:
        doc, _, clause = g.partition(":")
        parts = [b["text"] for b in blocks if b["doc"] == doc and
                 (b["clause"] == clause or b["clause"].startswith(clause + ".") or b["clause"].startswith(clause + "("))]
        out.append(f"[{g}]\n" + "\n".join(parts))
    return "\n\n".join(out)


def judge(llm: ClaudeClient, question, answer_text: str, blocks: list[dict]) -> dict:
    reference = gold_text(question.gold, blocks) if question.gold else "(The documents do not cover this question.)"
    msg = f"Question: {question.question}\n\nReference text:\n{reference}\n\nAssistant's answer:\n{answer_text}"
    resp = llm.create(JUDGE_SYSTEM, [{"role": "user", "content": msg}], tools=[GRADE_TOOL], max_tokens=300)
    out = next(dict(b.input) for b in resp.content if b.type == "tool_use")
    return {"label": out["label"], "reason": out["reason"],
            "tokens": resp.usage.input_tokens + resp.usage.output_tokens}


def score(rows: list[dict]) -> dict:
    ans = [r for r in rows if r["type"] == "answerable"]
    una = [r for r in rows if r["type"] == "unanswerable"]
    answered_ans = [r for r in ans if not r["refused"]]
    cited = [r for r in answered_ans if r["citations"]]

    def pct(xs, n):
        return round(sum(xs) / n, 3) if n else None

    return {
        "questions": len(rows),
        "accuracy": pct([r["judge"] == "correct" for r in ans], len(ans)),
        "accuracy_incl_partial": pct([r["judge"] in ("correct", "partial") for r in ans], len(ans)),
        "fact_check_pass": pct([r["facts_ok"] for r in ans], len(ans)),
        "judge_fact_agreement": pct([(r["judge"] == "correct") == r["facts_ok"] for r in ans], len(ans)),
        "wrongly_refused": pct([r["refused"] for r in ans], len(ans)),
        "abstained_on_unanswerable": pct([r["judge"] == "abstained" for r in una], len(una)),
        "citation_hit": pct([r["citation_hit"] for r in cited], len(cited)),
        "answers_with_citations": pct([bool(r["citations"]) for r in answered_ans], len(answered_ans)),
        "needed_retry": pct([r["attempts"] > 1 for r in rows], len(rows)),
        "unverified_fallbacks": pct([not r["verified"] and r["mode"] == "grounded" for r in rows], len(rows)),
        "avg_tokens": round(sum(r["tokens"] for r in rows) / len(rows)),
        "avg_seconds": round(sum(r["seconds"] for r in rows) / len(rows), 2),
        "by_kind": {k: pct([r["judge"] == "correct" for r in ans if r["kind"] == k],
                           len([r for r in ans if r["kind"] == k])) for k in sorted({r["kind"] for r in ans})},
    }


def run_system(name: str, questions, blocks, embedder, model: str | None, judge_llm) -> dict:
    chunking, retriever, mode = SYSTEMS[name]
    chunks = build_chunks(chunking)
    answerer = Answerer(build_retriever(retriever, chunks, embedder), ClaudeClient(model), mode=mode)
    by_id = {c.id: c for c in chunks}
    rows = []
    for i, q in enumerate(questions, 1):
        a = answerer.answer(q.question)
        g = judge(judge_llm, q, a.answer, blocks)
        cited_chunks = [by_id[c["chunk_id"]] for c in a.citations if c.get("chunk_id") in by_id]
        row = {
            "id": q.id, "type": q.type, "kind": q.kind, "mode": mode,
            "answer": a.answer, "refused": a.refused if mode == "grounded" else g["label"] == "abstained",
            "citations": a.citations, "retrieved": a.retrieved,
            "facts_ok": facts_present(a.answer, q.facts) if q.type == "answerable" else None,
            "citation_hit": any(c.covers(gl) for c in cited_chunks for gl in q.gold) if q.gold else None,
            "judge": g["label"], "judge_reason": g["reason"],
            "verified": a.verified, "attempts": a.attempts, "problems": a.problems,
            "tokens": a.input_tokens + a.output_tokens, "seconds": a.seconds,
        }
        rows.append(row)
        print(f"[{name} {i:>2}/{len(questions)}] {q.id:<5} {g['label']:<9} refused={row['refused']!s:<5} "
              f"facts={row['facts_ok']!s:<5} tries={a.attempts}")
    return {"system": name, "config": {"chunking": chunking, "retriever": retriever, "mode": mode,
                                       "model": answerer.llm.model, "judge": JUDGE_MODEL},
            "metrics": score(rows), "rows": rows}


def main():
    from grounded.embed import CachedEmbedder, Embedder

    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", nargs="+", default=list(SYSTEMS), choices=list(SYSTEMS))
    ap.add_argument("--model", default=None)
    ap.add_argument("--limit", type=int, default=None, help="only the first N questions (for a quick check)")
    args = ap.parse_args()

    questions = load_testset()[: args.limit]
    blocks = load_blocks()
    embedder = CachedEmbedder(Embedder())
    judge_llm = ClaudeClient(JUDGE_MODEL)
    config.REPORTS_DIR.mkdir(exist_ok=True)
    summary = {}
    for name in args.systems:
        t = time.time()
        result = run_system(name, questions, blocks, embedder, args.model, judge_llm)
        result["minutes"] = round((time.time() - t) / 60, 1)
        suffix = "" if args.limit is None else f"_first{args.limit}"
        (config.REPORTS_DIR / f"answers_{name}{suffix}.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
        summary[name] = result["metrics"]
        print(json.dumps({name: result["metrics"]}, indent=2))
    if args.limit is None:
        (config.REPORTS_DIR / "answers_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
