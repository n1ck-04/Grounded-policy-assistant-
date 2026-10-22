# Code walkthrough

Read the files in this order. It follows one question through the system, from saved web page to verified answer, then covers how the system is measured. Each section says what the file does, the parts worth understanding, and something to try.

About 1,260 lines of Python in `src/grounded/`, plus 21 tests.

---

## 0. `config.py`: every setting in one place

- **Paths** all hang off `ROOT`, which is the repo folder, or `GROUNDED_HOME` if set.
- **`SOURCES`**: the three documents. The key (`customer_liability`, ...) is the file name in `data/raw/` and the first part of every citation id.
- **Knobs**: `FIXED_CHUNK_WORDS = 200`, `FIXED_OVERLAP_WORDS = 40`, `CLAUSE_MAX_WORDS = 200`, `TOP_K = 5`, `RRF_K = 60`.
- **`LLM_MODEL`**: `claude-sonnet-5-5`, overridable with the `GROUNDED_MODEL` environment variable.

If you change a knob, re-run `make retrieval-eval` (free) and see what moves.

---

## 1. `ingest.py`: web page → clause-numbered blocks

**Input:** `data/raw/<doc>.html`. **Output:** `data/corpus/sections.jsonl`, one JSON line per block:

```json
{"doc": "customer_liability", "order": 31, "clause": "8", "kind": "table",
 "heading": "Reporting of unauthorised transactions ...", "text": "Within 3 working days | Zero liability\n..."}
```

**How `parse_html` works:**
1. `soup.select_one("table.tablebg")` finds the document body. RBI pages wrap the content in that table; everything outside it is site navigation.
2. Deletes noise: `<sup>` footnote markers, the page header cell (`td.tableheader`), scripts.
3. `_raw_blocks()` walks the body in reading order and yields `(kind, text)`:
   - `<p class="head">` → heading; `<p class="footnote">` → skipped.
   - Tables → `_table_text()`, one line per row, cells joined with ` | `.
   - Tables starting with "Index"/"Contents" → skipped (tables of contents would otherwise match every query).
   - Wrapper tags (`div`, `ol`, `td`, ...) → recurse into them.
4. Assigns a **clause id** to each block, using three regexes:
   - `NUMBERED` matches `2.1.1.2 ` at the start of a paragraph or heading → that's the clause.
   - `ROMAN_ITEM` matches `i.` or `(ii)` → becomes `top(i)`, a sub-clause of the current top-level clause (`10(i)`).
   - `ANNEX` → `_annex_id()` normalises "Annex II: List of..." to `Annex II`.
   - Anything else inherits the current clause.
5. **The reverse pass.** An unnumbered heading like "Reversal Timeline for Zero Liability" *introduces* the next clause, but the forward pass gave it the previous one. Walking backwards, each such heading takes the clause of the block after it.

**Try it:** `make ingest`, then search `data/corpus/sections.jsonl` for `"clause": "8"` in customer_liability and compare with the web page.

**Interview point:** "Why parse HTML structure instead of extracting plain text?" Because the clause ids are what make citations checkable and let the evaluation test retrieval against the *right clause*, not just similar words.

---

## 2. `chunking.py`: blocks → searchable chunks

A `Chunk` has an `id`, `doc`, the list of `clauses` it contains, a `heading` and the `text` that gets embedded and shown to the model.

**`Chunk.covers(gold)`** is used everywhere in the evaluation. A chunk covers `fraud_risk:3.3` if it contains clause `3.3` or any sub-clause (`3.3.4`, `3.3(i)`). Note the `+ "."` / `+ "("`: without them, clause `1` would match `10`.

**`clause_chunks()`**, the strategy the final system uses:
1. `group_key()` maps each clause to its group: `10(ii)` → `10`, `2.1.1.2` → `2.1`. Consecutive blocks with the same key form a group.
2. Groups made only of headings ("CHAPTER III") are held in `pending` and attached to the next group.
3. A group over 200 words is split at block boundaries, never mid-sentence. Parts get ids `doc:key#1`, `#2`.
4. The chunk text starts with a header line, `"<document name> | <most specific heading>"`, so a chunk makes sense on its own to both the embedding model and Claude.

**`fixed_chunks()`**, the comparison strategy: all the words of a document in order, cut into 200-word windows every 160 words. Each word remembers its clause (`owner` list), so a window still knows which clauses it touches, and the two strategies can be evaluated the same way.

Result on the three documents: 108 clause chunks, 81 fixed windows.

**Try it:**
```python
from grounded.chunking import build_chunks
cs = build_chunks("clause")
print(next(c for c in cs if c.id == "customer_liability:8").text)
```

---

## 3. `embed.py`: text → vectors

`Embedder` runs **bge-small-en-v1.5** with ONNX Runtime:
1. `BertWordPieceTokenizer(vocab.txt, lowercase=True)`, truncated to 512 tokens and padded per batch.
2. The model returns one 384-number vector per token; `hidden[:, 0]` takes the first one (`[CLS]`), which is how bge was trained to represent a whole text.
3. Each vector is divided by its length, so a dot product **is** cosine similarity.
4. `embed_query` adds `BGE_QUERY_PREFIX` first. Documents get no prefix. That asymmetry is part of how bge was trained, and leaving it out lowers retrieval quality.

`CachedEmbedder` stores document vectors in `data/cache/docs-<hash>.npy`. The hash is over all the chunk texts, so changing the chunking automatically creates a new cache entry.

**Interview point:** "Why not an embedding API?" No key or cost for search, it runs in CI-sized environments, and ONNX keeps the install small. The trade-off: a larger model might retrieve better. That's measurable with this harness.

---

## 4. `retrieve.py`: question → top 5 chunks

All retrievers have the same `search(query, k) -> [(chunk, score), ...]` interface, best first.

- **`tokenize()`**: lower-case words and numbers, with stopwords removed. `3.3.4` stays one token, so clause numbers can be searched.
- **`BM25Retriever`**: `rank_bm25.BM25Okapi`. Scores rise with how often query words appear in a chunk, weighted by how rare the word is across chunks.
- **`DenseRetriever`**: embeds all chunks once into `self.matrix` (108 × 384), then `matrix @ query_vector` gives every chunk's cosine similarity in one step.
- **`rrf_merge()`**: for each ranked list, a chunk at rank *r* gets `1 / (60 + r)`; scores add up across lists. A chunk ranked 3rd by both beats one ranked 1st by one and absent from the other.
- **`HybridRetriever`**: takes the top 20 from each, fuses them, returns the top 5.

**Try it:** compare the three on the hardest question:
```python
from grounded.chunking import build_chunks
from grounded.embed import CachedEmbedder, Embedder
from grounded.retrieve import build_retriever
cs, emb = build_chunks("clause"), CachedEmbedder(Embedder())
q = "How soon after classifying a fraud must a bank file the Fraud Monitoring Return?"
for kind in ("bm25", "dense", "hybrid"):
    print(kind, [c.id for c, _ in build_retriever(kind, cs, emb).search(q)])
```
None of them finds clause 6.3.1 because the document only says "FMR". That's Exercise 1.

---

## 5. `answer.py`: chunks → answer with citations

**`Answerer.answer(question)`:**
1. Retrieves the top 5 chunks and formats them as `<excerpt id="customer_liability:8"> ... </excerpt>`.
2. **Naive mode** (baseline only): one call with `SYSTEM_NAIVE`, returns the plain text, done.
3. **Grounded mode:** calls Claude with `SYSTEM_GROUNDED` and the `submit_answer` tool. The tool's JSON schema *is* the output format: `answer`, `citations: [{chunk_id, quote}]`, `refused`.
4. The loop, at most 2 attempts:
   - No tool call (Claude replied in prose) → ask again.
   - `validate()` finds problems → send them back as a `tool_result` with `is_error: True` ("Rejected: citation 1: the quote is not an exact copy..."). The model sees exactly what to fix. The failed citations are saved in `result.rejected`.
   - Passes → keep the answer, stop.
5. The `for ... else` block runs only if the loop never hit `break`, meaning no attempt passed. The result becomes a refusal with `verified=False`. **An unverified answer is never returned.**

`ClaudeClient` is a thin wrapper around the Anthropic SDK with one `create()` method. Tests replace it with `FakeLLM`, which returns scripted responses.

**Why no forced tool call?** The API's `tool_choice` option can force a tool, but Claude Sonnet 5.5 doesn't support forcing one. The prompt says "Always respond by calling submit_answer", and step 4 handles the rare text reply.

**Try it:** `make ask Q="Can a loan app read my contact list?"`, then a question it shouldn't answer: `make ask Q="What is the repo rate?"`. Look at `refused`, `citations`, `attempts`.

---

## 6. `guardrails.py`: is each citation real?

**`validate(result, shown)`** returns a list of problems; empty means it passed.
- Answer text is empty.
- A refusal with citations (contradictory).
- A non-refusal without citations.
- A `chunk_id` that wasn't among the 5 shown (an invented source).
- A quote that isn't in the cited chunk.

**`quote_in_chunk()`** normalises both sides with `_norm()`: lower-case, curly → straight quotes, en/em dashes → `-`, `₹` → `rs`, whitespace collapsed. So formatting differences don't fail a real quote, but different words do. A quote may use `...` to join two pieces; each piece must appear.

**What it proves and doesn't:** a verified quote proves the cited text exists in a document the model was shown. It does *not* prove the answer follows from that text. That part is measured by the judge in the evaluation.

**Known weakness:** table rows contain ` | ` separators and inconsistent spacing (`State / Union Territory`), which the model may not copy exactly. That's the probable cause of the one fallback in the evaluation (Exercise 8).

---

## 7. `api.py`: the HTTP service

`create_app(answerer=None)` builds the FastAPI app. Passing an answerer is how tests inject the fake LLM. Without one, the real answerer is built lazily on the first `/ask`, so `/health` works even without the model files.

- `GET /health`, `GET /documents` (titles, dates, links to RBI).
- `POST /ask` with `{"question": "..."}`. Pydantic rejects questions under 3 or over 500 characters with a 422 before any API call is made.

**Try it:** `make api`, open http://127.0.0.1:8000/docs and use "Try it out".

---

## 8. `evaluation.py`: is the search finding the right clause?

- **`load_testset()`** reads `evals/testset.yaml` into `Question` objects.
- **`normalise()` / `facts_present()`**: the deterministic fact check. Each fact is a list of acceptable phrasings; an answer passes if it contains at least one phrasing of *every* fact, after normalising `₹5,000` → `rs 5000`, `%` → `per cent`, and so on.
- **`retrieval_scores()`**, over answerable questions only:
  - **recall@5**: share of a question's gold clauses that appear in the top 5, averaged.
  - **hit@1**: the first chunk contains a gold clause.
  - **MRR**: 1 / rank of the first chunk with a gold clause (1.0 = first, 0.5 = second, ...), averaged.
  - It also records misses, recall per question kind, and the average number of words sent to the model.
- **`main()`** runs all 6 combinations (2 chunkings × 3 retrievers) and writes `reports/retrieval.json` and `retrieval.png`.

---

## 9. `answer_eval.py`: is the final answer right?

**`SYSTEMS`** defines the three compared configurations as `(chunking, retriever, mode)` tuples. Adding a system is one line.

**`run_system()`**: for each question, answer it, then judge it, and save one row with the answer, citations, retrieved ids, fact check, judge label and reason, attempts, problems, rejected quotes, tokens and seconds. Every row is in `reports/answers_<system>.json`, so any number in the README can be traced to individual answers.

**`judge()`**: Claude Haiku 5.5 gets the question, the *full text of the gold clauses* (`gold_text()`, pulled from the parsed blocks) and the answer, and calls `grade` with a label:
- answerable questions: `correct`, `partial` or `incorrect`;
- unanswerable questions: `abstained` or `answered`.

It doesn't know which system wrote the answer. `max_tokens=2000` because Haiku 5.5 thinks before answering; with 300, it sometimes ran out mid-thought (see DECISIONS.md #10). If it replies in prose, a single clearly stated label is accepted. After 3 failures the row is `ungraded`, and that count is reported, never hidden.

**`score()`** turns rows into the README's numbers: accuracy (judged correct), declined out-of-scope questions, citation hit rate (a cited chunk covers a gold clause), judge–fact agreement, retries, fallbacks, tokens and seconds.

**`--rejudge`** re-grades the saved answers without generating new ones. Use it after changing the judge, so the answers being compared stay identical.

**Try it cheaply:** `python -m grounded.answer_eval --systems grounded-clause --limit 5` (about 15 API calls). Results go to `answers_grounded-clause_first5.json` and don't overwrite the full run.

---

## 10. Tests

`tests/conftest.py` provides:
- `blocks` / `clause_chunks`: parsed from `tests/fixtures/sample_rbi.html`, an invented page with RBI's HTML structure (numbered clauses, roman items, a table, a footnote, a contents table).
- `FakeEmbedder`: bag-of-words vectors, so dense and hybrid search run without the model.
- `FakeLLM`: returns a scripted list of `submit_answer` inputs (or text for naive mode) and records every call.

What they cover:
- `test_ingest_chunking.py`: clause ids (dotted numbers, roman items, annexes), removed noise (footnotes, markers, the contents table, the page header), flattened tables, unnumbered headings, `group_key()`, clause chunks with their header line, `covers()`, and fixed-window overlap.
- `test_retrieval_answer.py`: BM25, dense, hybrid, the exact RRF arithmetic, retrieval metrics, fact matching, quote matching, every `validate()` failure, and the three answer-loop paths: verified first time, retry then success, refusal after two failures.

`make test` runs them all in about a second, with no network.

---

## Where to look when something changes

| You want to... | Change | Then run |
|---|---|---|
| Add a document | `SOURCES` in `config.py`, save the page to `data/raw/` | `make ingest`, `make retrieval-eval` |
| Change chunk size or top-k | `config.py` | `make retrieval-eval`, then `make eval` |
| Change the answer prompt | `SYSTEM_GROUNDED` in `answer.py` | `make eval-quick`, then `make eval` |
| Loosen or tighten quote matching | `_norm()` in `guardrails.py` | `make test`, then `make eval` |
| Change the grading | `JUDGE_SYSTEM` in `answer_eval.py` | `python -m grounded.answer_eval --rejudge` |
| Add test questions | `evals/testset.yaml` | `make retrieval-eval`, `make eval` |
