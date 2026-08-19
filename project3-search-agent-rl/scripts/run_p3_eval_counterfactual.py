#!/usr/bin/env python3
"""Phase 4A diagnostic 2: counterfactual evidence injection (non-confirmatory).

Independent diagnostic entry point; the formal eval script
(run_p3_eval_vllm_official.py) is NOT modified. For each dev256 question and
each model, runs one of four fixed conditions:

  no-evidence : base input only (SEARCH_PROMPT_PREFIX + question, user-only),
                model generates directly
  real-top3   : assistant <search>QUESTION</search> + user <information> real Top-3 docs </information>
  oracle      : if any Top-10 doc contains a normalized answer alias, inject
                that doc; no-hit questions are marked and get no evidence
  shuffled    : deterministic permutation (i -> (i+17) mod 256) of real Top-3

All conditions share question order, tokenizer (Base), seed 0, greedy decode.
EM uses the same skyrl compute_score semantics as the formal line
(method=strict, format_score=0.0); compliance = generated text contains
<answer>. Results are exploratory, NOT confirmatory.

Input rendering parity with the formal line: base content is
SEARCH_PROMPT_PREFIX + question rendered as a single user message with
add_generation_prompt=True (imported from the vendored env so the prefix is
byte-identical by construction). Retrieval (question-as-query, Top-10) is
shared across runs and cached under diag_cache/ so all conditions and models
see byte-identical evidence docs.

--cache-only builds/refreshes the evidence cache on CPU and exits (no GPU).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

from agent_system.environments.env_package.search.envs import SEARCH_PROMPT_PREFIX

# vLLM V0 greedy engine config mirrors the formal line.
VLLM_DTYPE = "bfloat16"
VLLM_GPU_MEMORY_UTILIZATION = 0.6
VLLM_MAX_MODEL_LEN = 2304  # max_input_tokens(2048) + max_new_tokens(256)

RETRIEVE_URL = "http://127.0.0.1:18080/retrieve"
DATA_ROOT = Path("/media/imc/data/project3-search-agent-rl")
CACHE_DIR = DATA_ROOT / "diag_cache"
CACHE_FILE = CACHE_DIR / "dev256_top10_docs.json"
DEV256_SHA = "ffebf468e756a673da267f5830cfc67f2e9c4dc44ec41c979a389c1efebfff60"


def norm_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", str(s))
    s = s.casefold()
    s = re.sub(r"[\s\W_]+", "", s)
    return s


def alias_hit_in_docs(answers: list[str], docs: list[dict]) -> bool:
    norm_answers = [norm_text(a) for a in answers if a]
    if not norm_answers:
        return False
    blob = " ".join(d.get("contents", "") for d in docs)
    nblob = norm_text(blob)
    return any(a and a in nblob for a in norm_answers)


def retrieve(query: str, topk: int, timeout: float = 180.0) -> dict:
    payload = json.dumps({"query": query, "topk": topk, "return_scores": True}).encode()
    req = Request(RETRIEVE_URL, data=payload, headers={"Content-Type": "application/json"})
    try:
        with urlopen(req, timeout=timeout) as resp:
            body = json.load(resp)
        docs = body["result"][0]
        return {
            "ok": True,
            "docs": [
                {"id": d["document"]["id"], "contents": d["document"]["contents"], "score": d.get("score")}
                for d in docs
            ],
        }
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"{type(e).__name__}: {e}", "docs": []}


def build_evidence_cache(questions: list[str]) -> dict:
    """Retrieve Top-10 per question (question-as-query), cache across runs."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    questions_sha = hashlib.sha256("\n".join(questions).encode()).hexdigest()
    if CACHE_FILE.exists():
        cache = json.loads(CACHE_FILE.read_text())
        if (
            cache.get("n") == len(questions)
            and cache.get("dev256_sha") == DEV256_SHA
            and cache.get("questions_sha") == questions_sha
        ):
            print(f"evidence cache hit: {CACHE_FILE} ({cache['n']} questions)")
            return cache
    entries = {}
    for i, q in enumerate(questions):
        res = retrieve(q, 10)
        entries[str(i)] = {"question": q, "ok": res["ok"], "docs": res.get("docs", []), "error": res.get("error")}
        sys.stdout.write(f"\rretrieving evidence cache {i + 1}/{len(questions)}")
        sys.stdout.flush()
    print()
    cache = {
        "dev256_sha": DEV256_SHA,
        "questions_sha": questions_sha,
        "n": len(questions),
        "entries": entries,
    }
    CACHE_FILE.write_text(json.dumps(cache, ensure_ascii=False))
    print(f"evidence cache written: {CACHE_FILE}")
    return cache


def render_chat(condition: str, question: str, docs_text: str, has_evidence: bool) -> list[dict]:
    """Fixed evidence template (prereg section 3): assistant <search> + user <information>."""
    msgs = [{"role": "user", "content": SEARCH_PROMPT_PREFIX + question}]
    if condition == "no-evidence":
        return msgs
    msgs.append({"role": "assistant", "content": f"<search>{question}</search>"})
    if not has_evidence:
        return msgs  # oracle no-hit or retrieval returned nothing; do not fabricate
    msgs.append({"role": "user", "content": "<information>\n" + docs_text + "\n</information>"})
    return msgs


def generate(llm, tokenizer, chats: list[list[dict]], args) -> list[str]:
    import vllm

    rendered = [
        tokenizer.apply_chat_template(c, add_generation_prompt=True, tokenize=False) for c in chats
    ]
    inputs = tokenizer(
        rendered,
        return_tensors=None,  # ragged lists: the vLLM engine batches internally
        padding=False,
        truncation=True,
        max_length=args.max_input_tokens,
    )
    sampling_params = vllm.SamplingParams(
        temperature=0.0, top_p=1.0, top_k=-1, max_tokens=args.max_new_tokens, ignore_eos=False
    )
    outputs = llm.generate(prompt_token_ids=inputs["input_ids"], sampling_params=sampling_params, use_tqdm=False)
    return [o.outputs[0].text for o in outputs]


def em_score(solution: str, answers: list[str]) -> float:
    from verl.utils.reward_score.search_r1_like_qa_em import compute_score

    return float(compute_score(solution, {"target": answers}, method="strict", format_score=0.0, score=1.0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--data-files", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--manifest-key", default="heldout")
    ap.add_argument("--condition", choices=["no-evidence", "real-top3", "oracle", "shuffled"])
    ap.add_argument("--max-input-tokens", type=int, default=2048)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--cache-only", action="store_true", help="build evidence cache on CPU and exit")
    args = ap.parse_args()

    # data SHA gate (same manifest mechanism as the formal line)
    sha = hashlib.sha256(args.data_files.read_bytes()).hexdigest()
    manifest = json.loads(args.manifest.read_text())
    expect = manifest["outputs"][args.manifest_key]["sha256"]
    assert sha == expect == DEV256_SHA, f"data SHA mismatch: got {sha}, manifest {expect}"

    df = pd.read_parquet(args.data_files)
    questions = [str(r["env_kwargs"]["question"]) for _, r in df.iterrows()]
    answers_list = [list(r["env_kwargs"]["ground_truth"]["target"]) for _, r in df.iterrows()]
    sources = list(df["data_source"])

    # shared evidence cache (Top-10 per question, question-as-query)
    cache = build_evidence_cache(questions)
    if args.cache_only:
        return
    entries = cache["entries"]

    # per-question docs for each condition: (docs_text, has_evidence)
    docs_plan: list[tuple[str, bool]] = []
    for i, q in enumerate(questions):
        if args.condition == "real-top3":
            docs = entries[str(i)]["docs"][:3]
            docs_plan.append(("\n".join(d["contents"] for d in docs), len(docs) > 0))
        elif args.condition == "oracle":
            docs10 = entries[str(i)]["docs"]
            hit = [d for d in docs10 if alias_hit_in_docs(answers_list[i], [d])]
            if hit:
                docs_plan.append((hit[0]["contents"], True))
            else:
                docs_plan.append(("", False))  # marked, no evidence injected
        elif args.condition == "shuffled":
            j = (i + 17) % len(questions)
            docs = entries[str(j)]["docs"][:3]
            docs_plan.append(("\n".join(d["contents"] for d in docs), len(docs) > 0))
        else:  # no-evidence
            docs_plan.append(("", False))

    chats = [
        render_chat(args.condition, questions[i], docs_plan[i][0], docs_plan[i][1]) for i in range(len(questions))
    ]

    import vllm
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm = vllm.LLM(
        model=str(args.model),
        tokenizer=str(args.tokenizer),
        dtype=VLLM_DTYPE,
        tensor_parallel_size=1,
        gpu_memory_utilization=VLLM_GPU_MEMORY_UTILIZATION,
        enforce_eager=True,
        max_model_len=VLLM_MAX_MODEL_LEN,
        seed=args.seed,
        trust_remote_code=False,
    )
    generated = generate(llm, tokenizer, chats, args)
    del llm

    episodes = []
    em_n = comp_n = 0
    per_source: dict[str, list[int]] = {}
    oracle_hit_em = [0, 0]  # [correct, total] for evidence-hit questions under oracle
    for i, (text, q, ans, src) in enumerate(zip(generated, questions, answers_list, sources)):
        em = em_score(text, ans)
        em_bool = em > 0.5  # compute_score returns 0.0 / 0.25(spam) / 1.0
        compliance = "<answer>" in text
        em_n += int(em_bool)
        comp_n += int(compliance)
        per_source.setdefault(src, [0, 0])[0] += int(em_bool)
        per_source[src][1] += 1
        if args.condition == "oracle" and docs_plan[i][1]:
            oracle_hit_em[1] += 1
            oracle_hit_em[0] += int(em_bool)
        episodes.append(
            {
                "question": q,
                "answers": ans,
                "source": src,
                "condition": args.condition,
                "evidence_injected": docs_plan[i][1],
                "oracle_hit": docs_plan[i][1] if args.condition == "oracle" else None,
                "generated": text,
                "em": float(em),
                "em_bool": bool(em_bool),
                "compliance": bool(compliance),
            }
        )

    n = len(episodes)
    results = {
        "condition": args.condition,
        "n": n,
        "em": em_n,
        "em_rate": em_n / n,
        "compliance": comp_n,
        "compliance_rate": comp_n / n,
        "per_source_em": {k: f"{v[0]}/{v[1]}" for k, v in sorted(per_source.items())},
        "oracle_evidence_hit_em": f"{oracle_hit_em[0]}/{oracle_hit_em[1]}" if args.condition == "oracle" else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        for ep in episodes:
            f.write(json.dumps(ep, ensure_ascii=False) + "\n")
    (args.output.parent / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
