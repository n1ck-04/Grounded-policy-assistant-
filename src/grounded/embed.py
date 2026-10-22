"""Text -> vectors with BAAI/bge-small-en-v1.5, run through ONNX Runtime.

Why this model
    Small (33M parameters, 384-dimensional vectors) and strong for its size on
    retrieval benchmarks, so it runs on an ordinary laptop CPU. MIT licensed.

Why ONNX Runtime instead of PyTorch / sentence-transformers
    The model only needs to run forwards. ONNX Runtime plus the `tokenizers`
    library is ~50 MB to install; PyTorch is over 700 MB. Anyone cloning the
    repo (a recruiter, CI) can run it quickly.

How a text becomes a vector
    1. WordPiece tokenizer (from vocab.txt, lower-cased, max 512 tokens).
    2. The model outputs one 384-number vector per token.
    3. bge uses the vector of the first token ([CLS]) as the text's embedding.
    4. Normalise to length 1, so a dot product equals cosine similarity.

    Queries get a short instruction prefix ("Represent this sentence for
    searching relevant passages: "); documents don't. bge was trained that way.

Embeddings of the chunks are cached on disk, keyed by a hash of the text, so
re-running the evaluation doesn't re-embed unchanged chunks.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from grounded import config


class Embedder:
    dim = 384

    def __init__(self, model_dir: Path = config.MODEL_DIR, batch_size: int = 16):
        import onnxruntime as ort
        from tokenizers import BertWordPieceTokenizer

        vocab, onnx = Path(model_dir) / "vocab.txt", Path(model_dir) / "onnx" / "model.onnx"
        if not onnx.exists():
            raise FileNotFoundError(f"{onnx} not found. See data/README.md for how to download the model.")
        self.tokenizer = BertWordPieceTokenizer(str(vocab), lowercase=True)
        self.tokenizer.enable_truncation(512)
        self.tokenizer.enable_padding()
        self.session = ort.InferenceSession(str(onnx), providers=["CPUExecutionProvider"])
        self.inputs = {i.name for i in self.session.get_inputs()}
        self.batch_size = batch_size

    def _encode(self, texts: list[str]) -> np.ndarray:
        out = []
        for i in range(0, len(texts), self.batch_size):
            enc = self.tokenizer.encode_batch(texts[i:i + self.batch_size])
            feed = {
                "input_ids": np.array([e.ids for e in enc], dtype=np.int64),
                "attention_mask": np.array([e.attention_mask for e in enc], dtype=np.int64),
                "token_type_ids": np.array([e.type_ids for e in enc], dtype=np.int64),
            }
            hidden = self.session.run(None, {k: v for k, v in feed.items() if k in self.inputs})[0]
            out.append(hidden[:, 0])                              # [CLS] vector
        vecs = np.vstack(out) if out else np.zeros((0, self.dim), dtype=np.float32)
        return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts)

    def embed_query(self, text: str) -> np.ndarray:
        return self._encode([config.BGE_QUERY_PREFIX + text])[0]


class CachedEmbedder:
    """Wraps an embedder; remembers document vectors on disk between runs."""

    def __init__(self, inner, cache_dir: Path = config.CACHE_DIR):
        self.inner, self.cache_dir = inner, Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        key = hashlib.sha256("\x00".join(texts).encode()).hexdigest()[:16]
        path = self.cache_dir / f"docs-{key}.npy"
        if path.exists():
            return np.load(path)
        vecs = self.inner.embed_documents(texts)
        np.save(path, vecs)
        return vecs

    def embed_query(self, text: str) -> np.ndarray:
        return self.inner.embed_query(text)
