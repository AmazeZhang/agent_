"""CPU-only tests for scripts/analyze_p3_eval_results.py.

Run with:  CUDA_VISIBLE_DEVICES='' python -m pytest -q tests/test_analyze_p3_eval.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from analyze_p3_eval_results import (
    build_report,
    classify_failure,
    exact_mcnemar,
    failure_classification,
    fmt_rate,
    load_run,
    matched_em_vectors,
    pairwise_mcnemar,
    render_markdown,
    wilson_interval,
    write_paired_csv,
)


# --------------------------------------------------------------------------
# fixtures: synthetic episodes following the run_p3_eval_heldout.py schema
# --------------------------------------------------------------------------

def make_step(*, status="success", executed_search=True, quality=None):
    quality = quality or {
        "has_search_tag": True, "has_answer_tag": False,
        "mixed_tags": False, "duplicate_tags": False, "projected_valid": True,
    }
    return {
        "step": 1, "prompt": "<|user|>q</user>", "raw_action": "<search>q</search>",
        "projected_action": ("search", {"query": "q"}),
        "action_quality": quality,
        "observation": {"text": "obs", "documents": ["d1"]},
        "reward": 0.0, "done": False, "won": False,
        "executed_search": executed_search,
        "info": {"tool_calling": executed_search,
                 "retrieval": {"status": status, "document_ids": ["d1"]}}
        if executed_search else {"tool_calling": False},
        "batch_generation_seconds": 1.0,
    }


def make_episode(question, source, *, em=True, offline_has_answer=True,
                 steps=None, reward=None, won=None):
    reward = (1.0 if reward is None else reward) if em else 0.0
    won = (True if won is None else won) if em else False
    steps = steps if steps is not None else []
    offline = {"final_answer": "<answer>a</answer>" if offline_has_answer else None,
               "score": reward, "has_answer": offline_has_answer}
    return {
        "question": question, "answers": ["ans"], "source": source, "steps": steps,
        "reward": reward, "done": True, "won": won, "offline": offline,
    }


def make_results(run_id, *, n=4, em=2, params=None, decoder="hf-transformers-greedy",
                 data_sha="deadbeef", adapter=None):
    params = params or {"seed": 0, "max_steps": 2, "history_length": 2, "topk": 3,
                        "timeout": 180, "max_input_tokens": 2048, "max_new_tokens": 256}
    adapter = adapter or {"path": "/none", "adapter_model.safetensors": "aa", "adapter_config.json": "bb"}
    return {
        "schema_version": 1, "kind": "p3-heldout-evaluation", "training": False,
        "training_operations": "none", "decoding_backend": decoder,
        "decoding_note": "HF greedy vs vLLM rollout differ", "retriever_note": "real",
        "run_id": run_id, "model_path": "/none", "adapter": adapter,
        "data_files": {"path": "/none", "sha256": data_sha,
                       "hash_verified_against_manifest": True},
        "leakage": {"overlap": 0}, "retriever_health": {"status": "ready", "vectors": 21015324},
        "parameters": params,
        "metrics": {"overall": {"n": n, "em": em, "success": em,
                                "em_rate": em / n, "success_rate": em / n,
                                "answer_compliance_rate": 1.0},
                    "per_source": {}, "action_stats": {}, "retrieval": {}},
        "outputs": {"episodes": "episodes.jsonl"},
    }


@pytest.fixture
def run_dirs(tmp_path):
    """Three run dirs: base (EM on q1,q3), step2 (EM on q2,q3), step5 (EM on q3,q4)."""
    questions = [f"q{i}" for i in range(1, 5)]
    em_by_model = {
        "base":  {1: True, 2: False, 3: True, 4: False},
        "step2": {1: False, 2: True, 3: True, 4: False},
        "step5": {1: False, 2: False, 3: True, 4: True},
    }
    dirs = []
    for label in ("base", "step2", "step5"):
        run_dir = tmp_path / f"run-{label}"
        run_dir.mkdir()
        episodes = []
        for qnum in range(1, 5):
            em = em_by_model[label][qnum]
            steps = [make_step()] if em else []
            episodes.append(make_episode(f"q{qnum}", "nq", em=em, steps=steps))
        (run_dir / "results.json").write_text(
            json.dumps(make_results(f"run-{label}"), ensure_ascii=False), encoding="utf-8")
        with open(run_dir / "episodes.jsonl", "w", encoding="utf-8") as handle:
            for ep in episodes:
                handle.write(json.dumps(ep, ensure_ascii=False) + "\n")
        dirs.append(run_dir)
    return dirs


# --------------------------------------------------------------------------
# Wilson interval
# --------------------------------------------------------------------------

def test_wilson_interval_bounds():
    lo, hi = wilson_interval(0, 32)
    assert lo == 0.0 and hi > 0.0
    lo, hi = wilson_interval(32, 32)
    assert hi == 1.0 and lo < 1.0
    lo, hi = wilson_interval(16, 32)
    assert lo <= 0.5 <= hi
    # n=32, k=16: Wilson 95% 宽度约 0.35
    assert 0.2 < (hi - lo) < 0.5
    assert wilson_interval(4, 0) == (0.0, 0.0)  # n=0 guard


# --------------------------------------------------------------------------
# exact McNemar
# --------------------------------------------------------------------------

def test_exact_mcnemar_symmetric_no_discordant():
    a = [True, False, True, False]
    res = exact_mcnemar(a, list(a))
    assert res["p_value"] == 1.0 and res["n_discordant"] == 0
    assert res["n11_both_right"] == 2 and res["n00_both_wrong"] == 2


def test_exact_mcnemar_extreme_one_sided():
    a = [False] * 10
    b = [False, False, False, False, False, True, True, True, True, True]
    res = exact_mcnemar(a, b)
    assert res["n01_second_only"] == 5 and res["n10_first_only"] == 0
    assert res["p_value"] == pytest.approx(2 * 0.5 ** 5)


def test_exact_mcnemar_balanced_discordant():
    # n01=2, n10=1 -> p = 2*(C(3,0)+C(3,1))*0.5^3 = 2*4/8 = 1.0 (cap)
    a = [True, False, False, False]
    b = [False, True, True, False]
    res = exact_mcnemar(a, b)
    assert res["n_discordant"] == 3
    assert res["p_value"] == pytest.approx(1.0)


def test_exact_mcnemar_length_mismatch():
    with pytest.raises(ValueError):
        exact_mcnemar([True], [True, False])


# --------------------------------------------------------------------------
# failure classification
# --------------------------------------------------------------------------

def test_classify_failure_categories():
    # answer format
    ep = make_episode("q", "nq", em=False, offline_has_answer=False)
    assert classify_failure(ep) == "answer_format"
    # invalid action (projected invalid)
    bad = dict(make_step(), action_quality={"has_search_tag": True, "has_answer_tag": False,
                                            "mixed_tags": False, "duplicate_tags": False,
                                            "projected_valid": False})
    ep = make_episode("q", "nq", em=False, steps=[bad])
    assert classify_failure(ep) == "invalid_action"
    # invalid action (mixed tags)
    mixed = dict(make_step(), action_quality={"has_search_tag": True, "has_answer_tag": True,
                                              "mixed_tags": True, "duplicate_tags": False,
                                              "projected_valid": True})
    ep = make_episode("q", "nq", em=False, steps=[mixed])
    assert classify_failure(ep) == "invalid_action"
    # no search
    ep = make_episode("q", "nq", em=False, steps=[make_step(executed_search=False)])
    assert classify_failure(ep) == "no_search"
    # retrieval failure
    ep = make_episode("q", "nq", em=False, steps=[make_step(status="invalid_query")])
    assert classify_failure(ep) == "retrieval_failure"
    # retrieved but wrong
    ep = make_episode("q", "nq", em=False, steps=[make_step(status="success")])
    assert classify_failure(ep) == "retrieved_but_wrong"
    # successes are excluded by failure_classification
    ep = make_episode("q", "nq", em=True, steps=[make_step()])
    assert failure_classification([ep]) == {}


# --------------------------------------------------------------------------
# matching and report end-to-end
# --------------------------------------------------------------------------

def test_matched_em_vectors_alignment(run_dirs):
    runs = [load_run(d) for d in run_dirs]
    vectors = matched_em_vectors({l: r["episodes"] for l, r in zip(("base", "step2", "step5"), runs)})
    assert vectors["base"] == [True, False, True, False]
    assert vectors["step5"] == [False, False, True, True]


def test_matched_em_vectors_mismatch_raises(tmp_path):
    d1 = tmp_path / "a"
    d1.mkdir()
    (d1 / "results.json").write_text(json.dumps(make_results("a")), encoding="utf-8")
    with open(d1 / "episodes.jsonl", "w") as h:
        h.write(json.dumps(make_episode("q1", "nq", em=True)) + "\n")
    d2 = tmp_path / "b"
    d2.mkdir()
    (d2 / "results.json").write_text(json.dumps(make_results("b")), encoding="utf-8")
    with open(d2 / "episodes.jsonl", "w") as h:
        h.write(json.dumps(make_episode("q1", "nq", em=True)) + "\n")
        h.write(json.dumps(make_episode("q2", "nq", em=False)) + "\n")
    runs = [load_run(d1), load_run(d2)]
    with pytest.raises(ValueError):
        matched_em_vectors({"a": runs[0]["episodes"], "b": runs[1]["episodes"]})


def test_build_report_end_to_end(run_dirs):
    report = build_report(run_dirs, ["base", "step2", "step5"])
    assert report["n_questions"] == 4
    assert report["parameters_consistent"] is True
    assert report["decoding_backends_consistent"] is True
    assert report["models"]["base"]["em"] == 2
    assert report["models"]["step5"]["em"] == 2
    # base 仅在 q1 对、step5 仅在 q4 对（各 1 个不一致对）-> 两尾精确 p = 1.0
    assert report["mcnemar"]["base<->step5"]["p_value"] == pytest.approx(1.0)
    assert report["mcnemar"]["base<->step5"]["n10_first_only"] == 1
    assert report["mcnemar"]["base<->step5"]["n01_second_only"] == 1
    assert report["models"]["base"]["artifacts"]["results.json"]
    assert report["models"]["base"]["artifacts"]["episodes.jsonl"]
    # sources present
    assert report["sources"] == ["nq"]


def test_build_report_parameter_inconsistency(run_dirs):
    params = {"seed": 0, "max_steps": 2, "history_length": 2, "topk": 9,  # different topk
              "timeout": 180, "max_input_tokens": 2048, "max_new_tokens": 256}
    run_dir = run_dirs[1]
    (run_dir / "results.json").write_text(
        json.dumps(make_results("run-step2", params=params)), encoding="utf-8")
    report = build_report(run_dirs, ["base", "step2", "step5"])
    assert report["parameters_consistent"] is False


def test_render_and_csv(run_dirs, tmp_path):
    report = build_report(run_dirs, ["base", "step2", "step5"])
    md = render_markdown(report)
    for section in ("总体指标", "分源 EM", "配对 McNemar", "失败案例分类", "证据与 hash", "声明边界"):
        assert section in md
    csv_path = tmp_path / "paired.csv"
    write_paired_csv(report, csv_path)
    rows = csv_path.read_text(encoding="utf-8").splitlines()
    assert rows[0] == "question,base_em,step2_em,step5_em"
    assert len(rows) == 5  # header + 4 questions


def test_fmt_rate_smoke():
    assert "50.0%" in fmt_rate(2, 4, (0.15, 0.85))
