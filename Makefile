# One command per step. No Make on Windows? Run the command after each colon directly.

.PHONY: install model ingest retrieval-eval eval eval-quick ask api test lint

install:          ## install the package and dev tools
	pip install -e ".[dev]" ruff

model:            ## download the embedding model (133 MB) into models/
	pip install "huggingface_hub[cli]"
	hf download BAAI/bge-small-en-v1.5 onnx/model.onnx vocab.txt --local-dir models/bge-small-en-v1.5

ingest:           ## parse the saved RBI pages in data/raw/ into clauses
	python -m grounded.ingest

retrieval-eval:   ## compare chunking x retrieval on the test set (free, ~20 s)
	python -m grounded.evaluation

eval:             ## end-to-end answer evaluation, 3 systems x 65 questions (needs API key)
	python -m grounded.answer_eval

eval-quick:       ## same, first 10 questions of the grounded system only
	python -m grounded.answer_eval --systems grounded-clause --limit 10

ask:              ## ask one question: make ask Q="your question"
	python -m grounded.answer "$(Q)"

api:              ## API at http://127.0.0.1:8000/docs
	uvicorn grounded.api:app --reload

test:             ## unit tests (no documents, model or API key needed)
	pytest -q

lint:
	ruff check src tests --select E,F,I,B --line-length 130
