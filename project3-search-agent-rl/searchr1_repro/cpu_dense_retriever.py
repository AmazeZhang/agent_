"""CPU-only E5 + FAISS retriever with random-access JSONL corpus storage."""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Protocol

import faiss
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModel, AutoTokenizer


class QueryEncoder(Protocol):
    def encode(self, query: str) -> np.ndarray: ...


class CorpusStore:
    def __init__(self, corpus_path: str | Path, offsets_path: str | Path):
        self.corpus_path = Path(corpus_path)
        self.offsets_path = Path(offsets_path)
        self.offsets = np.load(self.offsets_path, mmap_mode="r")
        if self.offsets.ndim != 1 or len(self.offsets) < 2:
            raise ValueError("corpus offsets must be a one-dimensional array with a sentinel")
        if int(self.offsets[0]) != 0 or int(self.offsets[-1]) != self.corpus_path.stat().st_size:
            raise ValueError("corpus offsets do not span the corpus file")
        self.rows = len(self.offsets) - 1
        self._fd = os.open(self.corpus_path, os.O_RDONLY)

    def get(self, index: int) -> dict:
        if index < 0 or index >= self.rows:
            raise IndexError(f"corpus index out of range: {index}")
        start = int(self.offsets[index])
        end = int(self.offsets[index + 1])
        record = json.loads(os.pread(self._fd, end - start, start))
        if record.get("id") != str(index):
            raise ValueError(f"corpus ID mismatch at index {index}: {record.get('id')!r}")
        return record

    def close(self) -> None:
        if getattr(self, "_fd", None) is not None:
            os.close(self._fd)
            self._fd = None

    def __del__(self):
        self.close()


class E5CpuEncoder:
    def __init__(self, model_path: str | Path):
        model_path = Path(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=True)
        self.model = AutoModel.from_pretrained(model_path, local_files_only=True).eval()

    @torch.inference_mode()
    def encode(self, query: str) -> np.ndarray:
        inputs = self.tokenizer(
            [f"query: {query}"],
            max_length=256,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        output = self.model(**inputs, return_dict=True)
        hidden = output.last_hidden_state.masked_fill(~inputs["attention_mask"][..., None].bool(), 0.0)
        embedding = hidden.sum(dim=1) / inputs["attention_mask"].sum(dim=1)[..., None]
        embedding = torch.nn.functional.normalize(embedding, dim=-1)
        return embedding.cpu().numpy().astype(np.float32, order="C")


class CpuDenseRetriever:
    def __init__(self, index, corpus: CorpusStore, encoder: QueryEncoder):
        self.index = index
        self.corpus = corpus
        self.encoder = encoder
        self._lock = threading.Lock()
        if self.index.ntotal != self.corpus.rows:
            raise ValueError(f"FAISS vectors ({self.index.ntotal}) != corpus rows ({self.corpus.rows})")

    @classmethod
    def load(cls, index_path: str | Path, corpus_path: str | Path, offsets_path: str | Path, model_path: str | Path):
        index = faiss.read_index(str(index_path))
        corpus = CorpusStore(corpus_path, offsets_path)
        encoder = E5CpuEncoder(model_path)
        return cls(index=index, corpus=corpus, encoder=encoder)

    def search(self, query: str, topk: int) -> list[dict]:
        with self._lock:
            embedding = self.encoder.encode(query)
            scores, indices = self.index.search(embedding, topk)
            return [
                {"document": self.corpus.get(int(index)), "score": float(score)}
                for index, score in zip(indices[0], scores[0])
                if index >= 0
            ]


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4096)
    topk: int | None = Field(default=None, ge=1, le=100)
    return_scores: bool = False


def create_app(retriever: CpuDenseRetriever, default_topk: int = 3, max_topk: int = 10) -> FastAPI:
    app = FastAPI()

    @app.get("/health")
    def health():
        return {
            "status": "ready",
            "index_class": type(retriever.index).__name__,
            "dimension": int(retriever.index.d),
            "vectors": int(retriever.index.ntotal),
            "corpus_rows": int(retriever.corpus.rows),
        }

    @app.post("/retrieve")
    def retrieve(request: QueryRequest):
        query = request.query.strip()
        if not query:
            raise HTTPException(status_code=422, detail="query must not be blank")
        topk = request.topk or default_topk
        if topk > max_topk:
            raise HTTPException(status_code=422, detail=f"topk exceeds service maximum {max_topk}")
        hits = retriever.search(query, topk)
        if request.return_scores:
            result = hits
        else:
            result = [hit["document"] for hit in hits]
        return {"result": [result]}

    return app
