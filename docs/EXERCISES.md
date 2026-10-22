# Exercises: extend the Grounded Policy Assistant

Each one changes real behaviour and has a measurable result. Work on a branch, commit with clear messages, and update the README if numbers change. Exercises marked 💳 call the Claude API; check the cost first with `--limit 10`.

---

## 1. Fix the hardest retrieval miss (1–2 h, free) ⭐

`fr10` ("How soon after classifying a fraud must a bank file the Fraud Monitoring Return?") is missed by every retriever. The document says **FMR**, the question says **Fraud Monitoring Return**.

1. Look at `reports/retrieval.json` → `rows` for `fr10`. Which chunks were retrieved instead?
2. Add a small acronym table to `retrieve.py` (FMR, SCBMF, CRILC, DLG, LSP, DLA, KFS, RE ...), built by reading the documents. Expand acronyms in both the query and the chunk text before BM25 tokenising.
3. Re-run `make retrieval-eval`. Did `fr10` move into the top 5? Did anything else get worse?

**Explain:** why does keyword search fail here, and why doesn't the meaning-based search save it?

## 2. Change how many chunks the model sees (1 h, free + 💳)

`TOP_K = 5` in `config.py`. Try 3 and 8 with `make retrieval-eval`, and plot recall@k.

Then run `python -m grounded.answer_eval --systems grounded-clause` for the best two values. More context raises recall, but does accuracy go up, or does the model get distracted? What happens to tokens per question?

## 3. Add a reranker (2–3 h, 💳)

Retrieve 20 candidates with hybrid search, then ask Claude Haiku to pick the 5 most relevant for the question. Add it as `RerankRetriever` in `retrieve.py`.

Compare recall@5 and hit@1 with plain hybrid, plus the extra cost and latency per question. Is it worth it here?

## 4. Add the missing documents (1–2 h, free)

Save one more RBI page (e.g. the Credit Card and Debit Card Master Direction) into `data/raw/`, add it to `SOURCES` in `config.py`, and run `make ingest`.

1. Does the parser handle its structure? Check `data/corpus/sections.jsonl`.
2. Write 5 new test questions for it, with gold clauses and facts.
3. Did adding a document make retrieval worse for the *existing* questions? More documents means more competition for the top 5.

## 5. Measure the judge (1 h, free)

The LLM judge marks answers correct/partial/incorrect. How far can it be trusted?

1. Pick 20 rows from `reports/answers_grounded-clause.json` and grade them yourself, without looking at the judge's label.
2. Compute your agreement with the judge, and list where you disagree and why.

This is the single most convincing thing you can say about an LLM evaluation in an interview: "I checked the judge against my own labels; we agreed on X of 20."

## 6. Stream answers in the API (2 h, 💳)

`POST /ask` waits for the whole answer. Add `POST /ask/stream` that streams the answer text with server-sent events, then sends the verified citations at the end. What can't you verify until the stream finishes, and how should the UI handle that?

## 7. A small web UI (2 h)

Build a single-page Streamlit app (`app.py`): a question box, the answer, and each citation shown as the exact quote under the clause id, with a link to the RBI page from `config.SOURCES`. Show refusals clearly, not as errors.

## 8. Fix the table-quote bug (1 h, free + 💳) ⭐

The evaluation found a real guardrail weakness. For `fr08` (private bank, ₹2 crore fraud: which agencies?), the answer is in a table, flattened as `₹1 crore and above | In addition to State/UT Police, ...`. The exact-quote check rejected Claude's citation twice, and the system refused a question it could answer.

1. Read `reports/answers_grounded-clause.json` → the `fr08` row → `problems`. The rejected quote wasn't saved in that run, so reproduce it: `make ask Q="..."` with the fr08 question, and look at the new `rejected` field.
2. Find the mismatch. Likely suspects: the `|` between cells, `State / Union Territory` vs `State/UT` spacing, or `₹`.
3. Write a failing test in `tests/test_retrieval_answer.py` with that quote and that chunk text.
4. Fix `_norm()` in `guardrails.py` (e.g. treat `|` as a space, and remove spaces around `/`). Make sure a quote with *different words* still fails.
5. Re-run `python -m grounded.answer_eval --systems grounded-clause` and update the README.

**Interview point:** the guardrail made the system refuse rather than show an unverifiable answer. The bug cost one answer, but never correctness. Explain why that's the right way for it to fail.
