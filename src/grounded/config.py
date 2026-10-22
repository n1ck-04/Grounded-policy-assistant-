"""Central settings: where things live, which documents, and the knobs we tune."""

import os
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get("GROUNDED_HOME") or (_REPO if (_REPO / "pyproject.toml").exists() else Path.cwd()))

RAW_DIR = ROOT / "data" / "raw"                      # the saved RBI pages (not in the repo; see data/README.md)
SECTIONS_PATH = ROOT / "data" / "corpus" / "sections.jsonl"
CACHE_DIR = ROOT / "data" / "cache"                  # cached embeddings
MODEL_DIR = ROOT / "models" / "bge-small-en-v1.5"    # embedding model (see data/README.md)
TESTSET_PATH = ROOT / "evals" / "testset.yaml"
REPORTS_DIR = ROOT / "reports"

# ---------------------------------------------------------------------------
# The documents. The id becomes part of every citation, e.g. customer_liability:7
# ---------------------------------------------------------------------------
SOURCES = {
    "customer_liability": {
        "title": "Customer Protection – Limiting Liability of Customers in Unauthorised Electronic Banking Transactions",
        "date": "2017-07-06",
        "url": "https://www.rbi.org.in/Scripts/NotificationUser.aspx?Id=11040",
    },
    "digital_lending": {
        "title": "Reserve Bank of India (Digital Lending) Directions, 2025",
        "date": "2025-05-08",
        "url": "https://www.rbi.org.in/Scripts/NotificationUser.aspx?Id=12848",
    },
    "fraud_risk": {
        "title": "Master Directions on Fraud Risk Management in Commercial Banks (incl. RRBs) and AIFIs",
        "date": "2024-07-15",
        "url": "https://www.rbi.org.in/Scripts/BS_ViewMasDirections.aspx?id=12702",
    },
}

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
FIXED_CHUNK_WORDS = 200         # "fixed" strategy: window size ...
FIXED_OVERLAP_WORDS = 40        # ... and overlap between neighbouring windows
CLAUSE_MAX_WORDS = 200          # "clause" strategy: split a clause group bigger than this, at clause boundaries
                                # (same size budget as "fixed", so the comparison is about boundaries, not size)

# ---------------------------------------------------------------------------
# Retrieval and answering
# ---------------------------------------------------------------------------
TOP_K = 5                       # chunks given to the model
RRF_K = 60                      # Reciprocal Rank Fusion constant (60 is the standard value)
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
LLM_MODEL = os.environ.get("GROUNDED_MODEL", "claude-sonnet-5-5")
