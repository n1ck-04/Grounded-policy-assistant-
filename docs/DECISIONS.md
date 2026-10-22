# Design decisions

Each entry: what was decided, what else was considered, and why. If you change one, update its entry.

---

### 1. RBI regulations as the domain

**Why:** banking rules are a real use case: customers and bank staff need exact answers ("how many days?", "who must approve?"), and a wrong answer has consequences. That makes grounding and refusals matter, not just fluency. The three documents fit together: lending rules, fraud handling, and customer liability for fraud. They also connect to my fraud-detection project.

### 2. Save the HTML pages instead of the PDFs

**Alternatives:** RBI's PDFs; scraping.

**Why:** RBI's PDF server refused direct downloads, but the notification pages contain the full text. HTML is actually better input: headings are marked `class="head"`, footnotes `class="footnote"`, and clause numbers are at the start of paragraphs. PDF extraction loses that structure and breaks tables. The pages aren't redistributed in the repo, because RBI's site reserves its rights; `data/README.md` explains how to save them.

### 3. Parse the clause structure

**Decision:** tag every block with its clause id (`2.1.1.2`, `10(ii)`, `Annex II`), flatten tables into `cell | cell` rows, and drop footnote markers, footnotes and tables of contents.

**Why:** citations can then name a clause ("fraud_risk:6.3.1"), and the evaluation can check the *right* clause was used. Without clause ids, you can only check that "some text" was retrieved.

**Details that mattered:**
- An unnumbered heading ("Reversal Timeline for…") belongs to the clause it *introduces*, so headings take the id of what follows.
- `i.` / `(ii)` items become sub-clauses of the clause they sit in (`10(i)`).
- A group that is only headings ("CHAPTER III") merges into the next chunk, so no chunk is just a title.

### 4. Two chunking strategies, compared at the same size

**Decision:** "fixed" (200-word windows, 40-word overlap) vs "clause" (clause groups, split only at clause boundaries, ≤200 words).

**What happened:** my first clause chunker allowed 350 words, and meaning-based search did worse with it (86% recall vs 97% for fixed windows). The cause: one chunk held all of clause 2.1, nine sub-clauses about notices, policy reviews and committees, so its embedding was a blur of nine topics. I then gave both strategies the same 200-word budget. That makes the comparison about *where* chunks break rather than *how big* they are, and it isn't tuned to the test questions.

**Result:**
- Fixed windows still had slightly higher retrieval recall (97% vs 94% with each's best search method).
- End to end, accuracy was identical (52/55), and clause chunks used 16% fewer tokens.
- The failures differed. Fixed windows failed twice because a window cut a clause in half; clause chunks failed once on retrieval.
- I kept clause chunks: equal accuracy, cheaper, and each citation points at one clause rather than a 200-word span of several.

### 5. bge-small-en-v1.5, run with ONNX Runtime

**Alternatives:** OpenAI/Voyage embedding APIs; larger open models; sentence-transformers on PyTorch.

**Why:**
- A 33M-parameter model with 384-dimensional vectors, strong for its size on retrieval benchmarks, MIT licensed. It runs on any laptop CPU (about 20 s to embed all chunks once; queries are instant).
- No API key or per-query cost for search.
- ONNX Runtime plus `tokenizers` is about 50 MB to install, versus over 700 MB for PyTorch.
- The tokenizer is rebuilt from `vocab.txt` (lower-cased WordPiece), so only two files are needed.

bge uses the `[CLS]` vector and a query prefix ("Represent this sentence for searching relevant passages: "), as it was trained.

### 6. Hybrid search with Reciprocal Rank Fusion

**Why:** keyword search finds exact terms ("SCBMF", "cooling-off", "120 days"), and meaning-based search finds paraphrases ("loan app" ≈ "DLA"). Each misses what the other catches. RRF merges the rankings using only rank positions, `1/(60 + rank)`, so BM25 scores and cosine similarities never need to be put on the same scale. With clause chunks, hybrid beat both components (93.6% vs 90.9% and 90.0%).

### 7. Structured output through a tool, with exact quotes

**Decision:** Claude must call `submit_answer(answer, citations[{chunk_id, quote}], refused)`.

**Why:**
- JSON every time, with nothing to parse out of prose.
- Exact quotes make citations *checkable*: a program can confirm the quote exists in the cited chunk.
- The `refused` flag makes "I don't know" a first-class, measurable outcome rather than something to detect in the text.

**A model detail:** Claude Sonnet 5.5 doesn't allow *forcing* a specific tool call, so the prompt says "always call submit_answer", and the loop treats a plain-text reply as a failed attempt.

### 8. Guardrails: verify, retry once, then refuse

**Decision:** check that cited ids were actually shown and that quotes match the chunk word for word (ignoring case, whitespace and quote styles; "…" can join pieces). On failure, send the specific problems back for one retry. If it fails again, refuse ("couldn't verify").

**Why:** the failure mode should be "no answer", never "unchecked answer". In the evaluation, the grounded-fixed system needed the retry twice (cl10, dl10) and passed both times. The grounded-clause system needed it once (fr08), failed again, and refused. The answer was in a table row, so the likely cause is the `|` separators and spacing that flattened tables contain. The rejected quote wasn't saved at the time; it is now (`Answer.rejected`). It's documented as Exercise 8 rather than silently fixed after seeing the test results.

### 9. A test set with refusal questions and gold clauses

**Decision:** 65 questions, written by reading the documents:
- 55 answerable, mostly in everyday words rather than the regulation's wording, some needing tables or two clauses.
- 10 that the documents don't cover. Five are "near misses" that sound like they should be covered (interest-rate caps, the Ombudsman's compensation limit); five are off-topic.

Each answerable question has gold clause ids and required facts.

**Why:** a test set that's all answerable can't measure the most important behaviour, which is declining. Paraphrased questions keep keyword search honest.

### 10. Two grading signals: fact check and LLM judge

**Why both:**
- *Deterministic facts* ("21 days", "5000") are cheap and reproducible, but miss correct paraphrases and can pass vague answers.
- *An LLM judge* (Claude Haiku 5.5) sees the question, the text of the gold clause and the answer. It catches paraphrases and wrong conditions, but it is itself a model.

Reporting their agreement (96% on grounded answers) tells the reader how much to trust the numbers. The judge never knows which system wrote the answer.

**A bug in the judge:** in the first run it returned no grade for 9 of 195 answers. Haiku 5.5 thinks before answering, and its 300-token limit ran out mid-thought. I raised the limit, re-graded *all* saved answers with the fixed judge (`--rejudge`), and kept the answers themselves unchanged.

### 11. A naive baseline, not just ablations

**Decision:** the same search, a plain "answer from the context" prompt, no citations, no refusal rule, no checks.

**Why:** it's what a first RAG prototype looks like, and it makes the value of grounding concrete. Accuracy was similar (92.7% vs 94.5%), but the naive system answered 2 of 10 out-of-scope questions from general knowledge and padded 2 others with claims not in the documents. The grounded systems did neither. "Similar accuracy, no unsupported answers" is the honest summary.

### 12. Tests that need nothing external

**Decision:** the unit tests use an invented page with RBI's structure (`tests/fixtures/sample_rbi.html`), a bag-of-words fake embedder, and a fake LLM that returns scripted tool calls.

**Why:** CI runs without the RBI pages (not redistributable), the 133 MB model, or an API key, and the tests are deterministic. The fake LLM makes it easy to test the paths that matter most: retry after a bad quote, and refusal after two failures.
