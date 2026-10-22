"""Retrieval, guardrails, the answer loop, evaluation helpers and the API."""

from fastapi.testclient import TestClient

from grounded.answer import Answerer
from grounded.api import create_app
from grounded.chunking import Chunk
from grounded.evaluation import Question, facts_present, normalise, retrieval_scores
from grounded.guardrails import quote_in_chunk, validate
from grounded.retrieve import BM25Retriever, DenseRetriever, HybridRetriever, rrf_merge, tokenize

from .conftest import FakeLLM


# ------------------------------------------------------------------------- retrieval
def test_tokenize_drops_stopwords_keeps_numbers():
    assert tokenize("What is the limit under clause 3.3.4?") == ["limit", "clause", "3.3.4"]


def test_bm25_finds_exact_terms(clause_chunks):
    top = BM25Retriever(clause_chunks).search("processing fee cooling-off", 1)[0][0]
    assert top.id == "sample:2"


def test_dense_and_hybrid_return_ranked_results(clause_chunks, fake_embedder):
    for r in (DenseRetriever(clause_chunks, fake_embedder), HybridRetriever(clause_chunks, fake_embedder)):
        res = r.search("zero liability when reporting fraud", 3)
        assert len(res) == 3 and res[0][1] >= res[1][1] >= res[2][1]


def test_rrf_rewards_agreement():
    a, b, c = (Chunk(i, "d", [], "", "") for i in "abc")
    merged = rrf_merge([[a, b, c], [b, a, c]])
    assert {merged[0][0].id, merged[1][0].id} == {"a", "b"} and merged[2][0].id == "c"
    assert abs(merged[0][1] - (1 / 61 + 1 / 62)) < 1e-12


def test_retrieval_scores(clause_chunks):
    qs = [Question("q1", "processing fee for exiting", "answerable", "lookup", ["sample:2"], [["fee"]]),
          Question("q2", "repo rate", "unanswerable", "off-topic", [], [])]
    s = retrieval_scores(BM25Retriever(clause_chunks), qs, k=3)
    assert s["recall@3"] == 1.0 and s["hit@1"] == 1.0 and len(s["rows"]) == 1   # unanswerable skipped


# ------------------------------------------------------------------------ evaluation
def test_normalise_and_facts():
    assert "5000" in normalise("₹5,000") and "5 per cent" in normalise("5%")
    assert facts_present("Liability is Rs. 5,000 or the transaction value.", [["5000"]])
    assert facts_present("Within three working days you owe nothing.", [["three working days", "3 working days"]])
    assert not facts_present("Within 10 days.", [["10 days"], ["home branch"]])


# ------------------------------------------------------------------------ guardrails
def test_quote_matching_is_forgiving_about_format_not_content():
    text = "3. A customer reporting fraud within two working days has  zero liability."
    assert quote_in_chunk("reporting fraud within two working days has zero liability", text)
    assert quote_in_chunk("A CUSTOMER reporting fraud ... zero liability", text)        # ellipsis joins pieces
    assert not quote_in_chunk("within three working days", text)                         # different content


def test_validate_catches_bad_citations(clause_chunks):
    shown = clause_chunks[:2]
    ok = {"answer": "x", "refused": False, "citations": [{"chunk_id": shown[0].id, "quote": shown[0].text.split("\n")[1][:30]}]}
    assert validate(ok, shown) == []
    fake_id = {**ok, "citations": [{"chunk_id": "made_up:9", "quote": "x"}]}
    assert any("not one of the excerpts" in p for p in validate(fake_id, shown))
    bad_quote = {**ok, "citations": [{"chunk_id": shown[0].id, "quote": "words that are not there"}]}
    assert any("exact copy" in p for p in validate(bad_quote, shown))
    assert validate({"answer": "y", "refused": False, "citations": []}, shown)          # no citation
    assert validate({"answer": "no", "refused": True, "citations": ok["citations"]}, shown)


# ---------------------------------------------------------------------- answer loop
def _good(chunk):
    return {"answer": "At least three days.", "refused": False,
            "citations": [{"chunk_id": chunk.id, "quote": "cooling-off period of not less than three days"}]}


def test_answer_verified_first_time(clause_chunks):
    r = BM25Retriever(clause_chunks)
    target = next(c for c in clause_chunks if c.id == "sample:2")
    a = Answerer(r, FakeLLM([_good(target)])).answer("How long is the cooling-off period?")
    assert a.verified and not a.refused and a.attempts == 1 and a.citations


def test_answer_retries_after_bad_quote_then_succeeds(clause_chunks):
    r = BM25Retriever(clause_chunks)
    target = next(c for c in clause_chunks if c.id == "sample:2")
    bad = {**_good(target), "citations": [{"chunk_id": target.id, "quote": "seven days"}]}
    llm = FakeLLM([bad, _good(target)])
    a = Answerer(r, llm).answer("How long is the cooling-off period?")
    assert a.attempts == 2 and a.verified and len(a.problems) == 1
    assert "Rejected" in str(llm.calls[1][-1]["content"])          # the model was told what failed


def test_answer_refuses_when_never_verifiable(clause_chunks):
    bad = {"answer": "Seven days.", "refused": False, "citations": [{"chunk_id": "nope:1", "quote": "x"}]}
    a = Answerer(BM25Retriever(clause_chunks), FakeLLM([bad, bad])).answer("cooling-off period?")
    assert a.refused and not a.verified and a.citations == []
    assert a.rejected == [bad["citations"], bad["citations"]]       # failed quotes kept for debugging


def test_naive_mode_returns_plain_text(clause_chunks):
    a = Answerer(BM25Retriever(clause_chunks), FakeLLM(["Three days."]), mode="naive").answer("cooling-off?")
    assert a.answer == "Three days." and not a.verified


# ------------------------------------------------------------------------------- api
def test_api(clause_chunks):
    target = next(c for c in clause_chunks if c.id == "sample:2")
    client = TestClient(create_app(Answerer(BM25Retriever(clause_chunks), FakeLLM([_good(target)]))))
    assert client.get("/health").json()["status"] == "ok"
    assert len(client.get("/documents").json()["documents"]) == 3
    r = client.post("/ask", json={"question": "How long is the cooling-off period?"})
    assert r.status_code == 200 and r.json()["citations"][0]["chunk_id"] == "sample:2"
    assert client.post("/ask", json={"question": "x"}).status_code == 422
