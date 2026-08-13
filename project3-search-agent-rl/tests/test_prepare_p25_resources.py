from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest

import scripts.prepare_p25_retriever_resources as prepare


def test_copy_and_hash():
    source = io.BytesIO(b"abc" * 100)
    destination = io.BytesIO()
    digest = hashlib.sha256()
    assert prepare.copy_and_hash(source, destination, digest) == 300
    assert destination.getvalue() == b"abc" * 100
    assert digest.hexdigest() == hashlib.sha256(b"abc" * 100).hexdigest()


def test_prepare_corpus_from_single_tar_member(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / "resources"
    archive_path = root / "corpus-download" / "wiki-18.jsonl.gz"
    archive_path.parent.mkdir(parents=True)
    payload = b'{"id":"0","contents":"title\\ntext"}\n{"id":"1","contents":"next"}\n'
    member_name = "nested/wiki_dump.jsonl"
    with tarfile.open(archive_path, "w:gz") as archive:
        info = tarfile.TarInfo(member_name)
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    monkeypatch.setattr(prepare, "CORPUS_MEMBER", member_name)
    monkeypatch.setattr(prepare, "CORPUS_MEMBER_SIZE", len(payload))
    monkeypatch.setattr(prepare, "CORPUS_ARCHIVE_SHA256", hashlib.sha256(archive_path.read_bytes()).hexdigest())
    prepared = root / "prepared"
    prepared.mkdir()
    result = prepare.prepare_corpus(root, prepared)
    assert result["rows"] == 2
    assert result["invalid_rows"] == 0
    assert result["observed_keys_first_100"] == ["contents", "id"]
    assert (prepared / "wiki-18.jsonl").read_bytes() == payload
    assert not (prepared / "wiki-18.jsonl.partial").exists()


def test_prepare_corpus_rejects_invalid_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / "resources"
    archive_path = root / "corpus-download" / "wiki-18.jsonl.gz"
    archive_path.parent.mkdir(parents=True)
    payload = b"not-json\n"
    with tarfile.open(archive_path, "w:gz") as archive:
        info = tarfile.TarInfo("corpus.jsonl")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    monkeypatch.setattr(prepare, "CORPUS_MEMBER", "corpus.jsonl")
    monkeypatch.setattr(prepare, "CORPUS_MEMBER_SIZE", len(payload))
    monkeypatch.setattr(prepare, "CORPUS_ARCHIVE_SHA256", hashlib.sha256(archive_path.read_bytes()).hexdigest())
    prepared = root / "prepared"
    prepared.mkdir()
    with pytest.raises(RuntimeError, match="invalid JSONL"):
        prepare.prepare_corpus(root, prepared)
