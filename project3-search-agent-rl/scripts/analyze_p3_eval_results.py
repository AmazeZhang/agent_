"""CPU-only comparison report for P3 held-out evaluation runs.

Reads three managed eval run dirs (Base / Step 2 / Step 5), each containing a
`results.json` and an `episodes.jsonl` produced by run_p3_eval_heldout.py, and
emits:

  - comparison.json      machine-readable report
  - comparison.md        human-readable report
  - paired_questions.csv per-question matched EM across the three models

Implements the report requirements of docs/P3_NEXT_ACTIONS_2026-08-14.md:
per-model overall/source metrics, binomial Wilson intervals, pairwise exact
McNemar tests on the same questions, failure-case classification, artifact
SHA256 evidence, and an HF-greedy vs vLLM-rollout backend note. No GPU, no
training code paths, no network access.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

WILSON_Z = 1.96

RETRIEVAL_FAILURE_STATUSES = {"invalid_query", "api_error", "processing_error", "no_results"}

FAILURE_CATEGORIES = {
    "answer_format": "答案格式（无 <answer> 或无法提取）",
    "invalid_action": "无效动作（投影失败 / 混合 / 重复标签）",
    "retrieval_failure": "检索失败（invalid_query / api_error / no_results）",
    "retrieved_but_wrong": "检索成功但答错",
    "no_search": "未搜索（直接作答）",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def wilson_interval(k: int, n: int, z: float = WILSON_Z) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (no continuity correction)."""
    if n <= 0:
        return (0.0, 0.0)
    phat = k / n
    denom = 1.0 + z * z / n
    centre = (phat + z * z / (2.0 * n)) / denom
    margin = z * math.sqrt(phat * (1.0 - phat) / n + z * z / (4.0 * n * n)) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def exact_mcnemar(a: list[bool], b: list[bool]) -> dict[str, Any]:
    """Two-sided exact McNemar test on matched binary outcomes.

    Discordant pairs (n01, n10) follow Binomial(n01+n10, 0.5) under the null;
    p = 2 * P(X <= min(n01, n10)), capped at 1.0.
    """
    if len(a) != len(b):
        raise ValueError(f"matched vectors differ in length: {len(a)} vs {len(b)}")
    n01 = n10 = n11 = n00 = 0
    for x, y in zip(a, b):
        if x and y:
            n11 += 1
        elif x and not y:
            n10 += 1
        elif not x and y:
            n01 += 1
        else:
            n00 += 1
    discordant = n01 + n10
    if discordant == 0:
        p_value = 1.0
    else:
        x = min(n01, n10)
        p_value = 2.0 * sum(math.comb(discordant, i) for i in range(x + 1)) * 0.5 ** discordant
        p_value = min(1.0, p_value)
    return {
        "p_value": p_value,
        "n00_both_wrong": n00,
        "n10_first_only": n10,
        "n01_second_only": n01,
        "n11_both_right": n11,
        "n_discordant": discordant,
    }


def load_run(run_dir: Path) -> dict[str, Any]:
    """Load results.json + episodes.jsonl; return a normalized run dict."""
    run_dir = Path(run_dir)
    results_path = run_dir / "results.json"
    episodes_path = run_dir / "episodes.jsonl"
    if not results_path.exists():
        raise FileNotFoundError(f"missing {results_path}")
    results = json.loads(results_path.read_text(encoding="utf-8"))
    episodes: list[dict[str, Any]] = []
    if episodes_path.exists():
        episodes = [json.loads(line) for line in episodes_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {
        "dir": run_dir,
        "results": results,
        "episodes": episodes,
        "episodes_path": episodes_path,
    }


def classify_failure(episode: dict[str, Any]) -> str:
    """Classify a non-EM episode into one of the five failure categories."""
    offline = episode.get("offline", {})
    if not offline.get("has_answer"):
        return "answer_format"
    steps: list[dict[str, Any]] = episode.get("steps", [])
    qualities = [s.get("action_quality", {}) for s in steps]
    if any(q.get("projected_valid") is False for q in qualities):
        return "invalid_action"
    if any(q.get("mixed_tags") or q.get("duplicate_tags") for q in qualities):
        return "invalid_action"
    searched = [s for s in steps if s.get("executed_search")]
    if not searched:
        return "no_search"
    statuses = [
        s.get("info", {}).get("retrieval", {}).get("status")
        for s in steps
        if s.get("info", {}).get("retrieval", {}).get("status")
    ]
    if statuses and all(st in RETRIEVAL_FAILURE_STATUSES for st in statuses):
        return "retrieval_failure"
    return "retrieved_but_wrong"


def failure_classification(episodes: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for episode in episodes:
        if episode.get("reward", 0.0) >= 1.0:
            continue  # success, not classified
        counts[classify_failure(episode)] += 1
    return dict(sorted(counts.items()))


def per_source_summary(episodes: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for episode in episodes:
        buckets.setdefault(episode["source"], []).append(episode)
    out: dict[str, dict[str, Any]] = {}
    for source, rows in sorted(buckets.items()):
        n = len(rows)
        em = sum(1 for r in rows if r.get("reward", 0.0) >= 1.0)
        success = sum(1 for r in rows if r.get("won"))
        lo, hi = wilson_interval(em, n)
        out[source] = {"n": n, "em": em, "em_rate": em / n if n else 0.0,
                       "em_ci_95": [round(lo, 4), round(hi, 4)],
                       "success": success, "success_rate": success / n if n else 0.0}
    return out


def matched_em_vectors(episodes_by_model: dict[str, list[dict[str, Any]]]) -> dict[str, list[bool]]:
    """Align episodes across models by question text and return EM vectors."""
    first_model = next(iter(episodes_by_model))
    questions = [e["question"] for e in episodes_by_model[first_model]]
    vectors: dict[str, list[bool]] = {}
    for label, episodes in episodes_by_model.items():
        by_question = {e["question"]: bool(e.get("reward", 0.0) >= 1.0) for e in episodes}
        if set(by_question) != set(questions):
            raise ValueError(f"run {label}: question set differs from {first_model} "
                             f"(only_here={len(set(by_question) - set(questions))}, "
                             f"missing={len(set(questions) - set(by_question))})")
        vectors[label] = [by_question[q] for q in questions]
    return vectors


def pairwise_mcnemar(vectors: dict[str, list[bool]]) -> dict[str, dict[str, Any]]:
    labels = list(vectors)
    out: dict[str, dict[str, Any]] = {}
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            key = f"{labels[i]}<->{labels[j]}"
            out[key] = exact_mcnemar(vectors[labels[i]], vectors[labels[j]])
    return out


def fmt_rate(em: int, n: int, ci: tuple[float, float]) -> str:
    return f"{em}/{n} ({em / n * 100:.1f}%) [95% CI {ci[0] * 100:.1f}–{ci[1] * 100:.1f}]"


def build_report(run_dirs: list[Path], labels: list[str]) -> dict[str, Any]:
    if len(run_dirs) != len(labels):
        raise ValueError("--runs and --labels must match in count")
    runs = [load_run(d) for d in run_dirs]

    # Parameter consistency audit across runs.
    parameter_sets = [json.dumps(r["results"].get("parameters", {}), sort_keys=True) for r in runs]
    parameters_consistent = len(set(parameter_sets)) == 1
    backends = {r["results"].get("decoding_backend") for r in runs}
    backends_consistent = len(backends) == 1

    per_model: dict[str, dict[str, Any]] = {}
    for label, run in zip(labels, runs):
        results = run["results"]
        metrics = results["metrics"]
        overall = metrics["overall"]
        episodes = run["episodes"]
        per_model[label] = {
            "run_id": results.get("run_id"),
            "dir": str(run["dir"]),
            "n": overall["n"],
            "em": overall["em"],
            "em_rate": overall["em_rate"],
            "em_ci_95": list(wilson_interval(overall["em"], overall["n"])),
            "success": overall["success"],
            "success_rate": overall["success_rate"],
            "success_ci_95": list(wilson_interval(overall["success"], overall["n"])),
            "answer_compliance_rate": overall.get("answer_compliance_rate", 0.0),
            "per_source": per_source_summary(episodes),
            "failures": failure_classification(episodes),
            "action_stats": metrics.get("action_stats", {}),
            "retrieval": metrics.get("retrieval", {}),
            "elapsed_seconds": results.get("elapsed_seconds"),
            "peak_gpu_memory_allocated_bytes": results.get("peak_gpu_memory_allocated_bytes"),
            "adapter": results.get("adapter"),
            "data_files": results.get("data_files"),
            "retriever_health": results.get("retriever_health"),
            "artifacts": {
                "results.json": sha256_file(run["dir"] / "results.json"),
                "episodes.jsonl": sha256_file(run["episodes_path"]) if run["episodes_path"].exists() else None,
            },
        }

    episodes_by_model = {label: run["episodes"] for label, run in zip(labels, runs)}
    vectors = matched_em_vectors(episodes_by_model)
    mcnemar = pairwise_mcnemar(vectors)
    questions = [e["question"] for e in episodes_by_model[labels[0]]]

    return {
        "models": per_model,
        "labels": labels,
        "n_questions": len(questions),
        "parameters_consistent": parameters_consistent,
        "parameters": runs[0]["results"].get("parameters", {}),
        "decoding_backend": sorted(backends) if backends_consistent else sorted(backends),
        "decoding_backends_consistent": backends_consistent,
        "decoding_note": runs[0]["results"].get("decoding_note"),
        "retriever_note": runs[0]["results"].get("retriever_note"),
        "mcnemar": mcnemar,
        "matched_em": vectors,
        "questions": questions,
        "sources": sorted({e["source"] for e in episodes_by_model[labels[0]]}),
    }


def render_markdown(report: dict[str, Any]) -> str:
    labels = report["labels"]
    lines: list[str] = []
    lines.append("# P3 Held-out 评测对比报告")
    lines.append("")
    lines.append("| Run | 模型 | run_id |")
    lines.append("|---|---|---|")
    for label in labels:
        m = report["models"][label]
        lines.append(f"| [{label}]({m['dir']}) | {m['run_id']} | `{m['run_id']}` |")
    lines.append("")
    lines.append(f"- 评测题数：{report['n_questions']}（同一批问题逐题配对）")
    lines.append(f"- 参数一致性：{'一致 ✓' if report['parameters_consistent'] else '**不一致！** ' + json.dumps(report['parameters'], ensure_ascii=False)}")
    lines.append(f"- 解码后端：{report['decoding_backend']}；HF 与 vLLM 一致：{'是' if report['decoding_backends_consistent'] else '否'}")
    if report["decoding_note"]:
        lines.append(f"- 解码说明：{report['decoding_note']}")
    lines.append("")
    lines.append("## 总体指标（含 Wilson 95% 区间）")
    lines.append("")
    lines.append("| 模型 | EM | success | answer 合规率 | 无效动作率 | invalid query 率 | api error 率 |")
    lines.append("|---|---|---|---|---|---|---|")
    for label in labels:
        m = report["models"][label]
        ov = f"{m['em']}/{m['n']} ({m['em_rate'] * 100:.1f}%) [95% CI {m['em_ci_95'][0] * 100:.1f}–{m['em_ci_95'][1] * 100:.1f}]"
        sv = f"{m['success']}/{m['n']} ({m['success_rate'] * 100:.1f}%) [95% CI {m['success_ci_95'][0] * 100:.1f}–{m['success_ci_95'][1] * 100:.1f}]"
        retr = m.get("retrieval", {})
        act = m.get("action_stats", {})
        lines.append(f"| {label} | {ov} | {sv} | {m['answer_compliance_rate'] * 100:.1f}% | "
                     f"{act.get('invalid_action_ratio', 0.0) * 100:.1f}% | "
                     f"{retr.get('invalid_query_rate', 0.0) * 100:.1f}% | "
                     f"{retr.get('api_error_rate', 0.0) * 100:.1f}% |")
    lines.append("")
    lines.append("## 分源 EM")
    lines.append("")
    header = "| 来源 | " + " | ".join(f"{l} (n, EM率)" for l in labels) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (len(labels) + 1))
    for source in report["sources"]:
        cells = [source]
        for label in labels:
            ps = report["models"][label]["per_source"].get(source, {})
            cells.append(f"{ps.get('n', 0)}条 {ps.get('em', 0)} ({ps.get('em_rate', 0.0) * 100:.1f}%)")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## 配对 McNemar（同一问题逐题对比，两尾精确检验）")
    lines.append("")
    lines.append("| 对比 | 都错 | 仅前者对 | 仅后者对 | 都对 | 不一致对 | p 值 |")
    lines.append("|---|---|---|---|---|---|---|")
    for key, res in report["mcnemar"].items():
        a, b = key.split("<->")
        lines.append(f"| {a} ↔ {b} | {res['n00_both_wrong']} | {res['n10_first_only']} | "
                     f"{res['n01_second_only']} | {res['n11_both_right']} | {res['n_discordant']} | "
                     f"{res['p_value']:.4f} |")
    lines.append("")
    lines.append("## 失败案例分类（EM=0 的 episodes）")
    lines.append("")
    lines.append("| 类别 | " + " | ".join(labels) + " |")
    lines.append("|" + "---|" * (len(labels) + 1))
    for category, zh in FAILURE_CATEGORIES.items():
        cells = [f"{category}（{zh}）"]
        for label in labels:
            cells.append(str(report["models"][label]["failures"].get(category, 0)))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## 证据与 hash 归档")
    lines.append("")
    lines.append("| Run | results.json SHA256 | episodes.jsonl SHA256 | 数据文件 SHA256 | 核对 |")
    lines.append("|---|---|---|---|---|")
    for label in labels:
        m = report["models"][label]
        data = m.get("data_files") or {}
        lines.append(f"| {label} | `{m['artifacts']['results.json']}` | "
                     f"`{m['artifacts']['episodes.jsonl'] or '缺失'}` | "
                     f"`{data.get('sha256', '?')}` | {data.get('hash_verified_against_manifest', '?')} |")
    lines.append("")
    lines.append("## 声明边界")
    lines.append("")
    lines.append("- 本报告只对评测 run 做机械汇总；样本量（n≤32）只能作为小样本初步证据。")
    lines.append("- 解码为 HF transformers 贪心（temperature 0），与训练期 vLLM rollout 存在 backend 差异；"
                 "若出现明确提升，必须先用 verl/vLLM 原生评测复核关键结论。")
    lines.append("- smoke-16 结果仅作管线门禁，不用于任何质量声明。")
    return "\n".join(lines)


def write_paired_csv(report: dict[str, Any], path: Path) -> None:
    labels = report["labels"]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["question"] + [f"{l}_em" for l in labels])
        for i, q in enumerate(report["questions"]):
            writer.writerow([q] + [1 if report["matched_em"][l][i] else 0 for l in labels])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True,
                        help="three managed eval run dirs (Base, Step 2, Step 5)")
    parser.add_argument("--labels", nargs="+", default=["base", "step2", "step5"])
    parser.add_argument("--out", type=Path, default=None, help="output dir (default: cwd)")
    args = parser.parse_args(argv)

    report = build_report([Path(d) for d in args.runs], args.labels)
    out_dir = args.out or Path.cwd()
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "comparison.json"
    md_path = out_dir / "comparison.md"
    csv_path = out_dir / "paired_questions.csv"

    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
    md_path.write_text(render_markdown(report), encoding="utf-8")
    write_paired_csv(report, csv_path)

    print(render_markdown(report))
    print(f"\nwritten: {json_path} {md_path} {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
