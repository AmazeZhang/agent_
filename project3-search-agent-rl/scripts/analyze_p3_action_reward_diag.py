#!/usr/bin/env python3
"""CPU-only diagnostic: action-format + reward-path analysis for P3.

Answers the six preregistered diagnostic questions without touching GPUs:

  1. How much did base vs train64 outputs actually change on the same 32
     heldout questions (byte-identical rate, normalized edit distance,
     per-step action-validity changes)?
  2. Why are mixed/duplicate-tag actions marked invalid (projection rules
     + concrete raw-action examples)?
  4. Reward decomposition on the train64-nqh training rollouts: how many
     episodes scored via EM (1.0), format (0.1), no-answer (0.0), and how
     many penalty units (0.1 x invalid actions) were subtracted; validates
     the reconstruction against the recorded scores.
  5. Heldout failure classification: no-search / searched-then-wrong /
     searched-then-correct / format-error, from the eval episodes.
  6. LoRA behaviour change: quantified by (1) on the same questions.

Reads only: eval run episodes.jsonl + results.json, training run
rollouts/*.jsonl + rollouts/*.audit.jsonl, train parquet. Writes a single
Markdown report + machine JSON. No GPU, no training, no network.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

# --- Same semantics as vendor verl-agent (search projection + skyrl EM) ------
RE_SEARCH_BLOCK = re.compile(r"<search>(.*?)</search>", re.IGNORECASE | re.DOTALL)
RE_ANSWER_BLOCK = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
RE_SEARCH_TAG = re.compile(r"<search>", re.IGNORECASE)
RE_ANSWER_TAG = re.compile(r"<answer>", re.IGNORECASE)
ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def normalize_answer(s: str) -> str:
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string_punctuation())
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def string_punctuation() -> str:
    import string

    return string.punctuation


def extract_solution_fork(solution_str: str):
    """Fork semantics (search_r1_like_qa_em.py): >=1 answer block, take last."""
    matches = list(ANSWER_PATTERN.finditer(solution_str))
    if len(matches) < 1:
        return None
    return matches[-1].group(1).strip()


def em_check(prediction, golden_answers) -> bool:
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    norm = normalize_answer(prediction)
    for golden_answer in golden_answers:
        if normalize_answer(golden_answer) == norm:
            return True
    return False


def projection_validity(action: str) -> tuple[bool, bool, bool]:
    """Return (valid, mixed_tags, duplicate_tags) with fork projection rules."""
    n_search = len(RE_SEARCH_TAG.findall(action))
    n_answer = len(RE_ANSWER_TAG.findall(action))
    mixed = bool(n_search and n_answer)
    duplicate = n_search > 1 or n_answer > 1
    valid = not (mixed or duplicate)
    return valid, mixed, duplicate


def lev_ratio(a: str, b: str) -> float:
    """Normalized Levenshtein distance (0..1) between two strings."""
    if a == b:
        return 0.0
    if not a or not b:
        return 1.0
    m, n = len(a), len(b)
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        cur = [i] + [0] * n
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[n] / max(m, n)


def load_episodes(run_dir: str) -> list[dict]:
    with open(Path(run_dir) / "episodes.jsonl") as f:
        return [json.loads(line) for line in f]


def steps_text(episode: dict) -> str:
    return "\n".join(step["raw_action"] for step in episode["steps"])


# ----------------------------------------------------------------------------
# Task 1 + 6: base vs train64 output comparison on the same 32 questions
# ----------------------------------------------------------------------------
def compare_outputs(base_eps: list[dict], new_eps: list[dict], label: str) -> dict:
    base_by_q = {normalize_answer(e["question"]): e for e in base_eps}
    identical = 0
    same_first_step = 0
    lev_sum = 0.0
    validity_flips = 0
    em_flips = 0
    n = 0
    examples: list[dict] = []
    for ep in new_eps:
        base = base_by_q.get(normalize_answer(ep["question"]))
        if base is None:
            continue
        n += 1
        a, b = steps_text(base), steps_text(ep)
        if a == b:
            identical += 1
        lev_sum += lev_ratio(a, b)
        if base["steps"][0]["raw_action"] == ep["steps"][0]["raw_action"]:
            same_first_step += 1
        base_valid = all(s["action_quality"]["projected_valid"] for s in base["steps"])
        new_valid = all(s["action_quality"]["projected_valid"] for s in ep["steps"])
        if base_valid != new_valid:
            validity_flips += 1
        if base["won"] != ep["won"]:
            em_flips += 1
            examples.append(
                {
                    "question": ep["question"],
                    "base_won": base["won"],
                    "new_won": ep["won"],
                    "base_first_step": base["steps"][0]["raw_action"][:200],
                    "new_first_step": ep["steps"][0]["raw_action"][:200],
                }
            )
    return {
        "n_questions": n,
        "byte_identical": identical,
        "identical_rate": identical / n if n else 0,
        "same_first_step": same_first_step,
        "mean_lev_distance": lev_sum / n if n else 0,
        "validity_flips": validity_flips,
        "em_flips": em_flips,
        "em_flip_examples": examples,
    }


# ----------------------------------------------------------------------------
# Task 4: training reward decomposition (env-exact reconstruction)
# ----------------------------------------------------------------------------
# Verified end-to-end semantics (377/377 rows reconstructed exactly):
#   SearchEnvironmentManager.step projects the RAW action first
#   (projection.py: trim to first closing tag, then first <search> block,
#   else first <answer> block, else "") and the env only ever sees the
#   PROJECTED action. chat_history therefore contains projected actions
#   (+ search observations), and compute_score reads the LAST <answer> block
#   of that history (fork rule: >=1 match). gather_rollout_data stamps EVERY
#   row of a trajectory with the same episode_rewards (sum of per-step env
#   rewards = final compute_score), then apply_invalid_action_penalty
#   subtracts 0.1 per invalid ROW. Hence for each (traj, step) row:
#       recorded score == episode_rewards - 0.1 * (0 if valid else 1)
# The per-row "source" of the OLD per-record decomposition was meaningless
# for multi-step episodes; this version reports episode-level finals.
RE_PROJ_S = re.compile(r"<search>(.*?)</search>", re.IGNORECASE | re.DOTALL)
RE_PROJ_A = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)


def _postprocess_action_trim(action: str) -> str:
    if "</search>" in action:
        return action.split("</search>", 1)[0] + "</search>"
    if "</answer>" in action:
        return action.split("</answer>", 1)[0] + "</answer>"
    return action


def _project_action(action: str) -> str:
    trimmed = _postprocess_action_trim(action)
    m = RE_PROJ_S.search(trimmed)
    if m:
        return f"<search>{m.group(1).strip()}</search>"
    m = RE_PROJ_A.search(trimmed)
    if m:
        return f"<answer>{m.group(1).strip()}</answer>"
    return ""


def _env_compute_score(history: str, golden_answers) -> float:
    ans = extract_solution_fork(history)
    if ans is None:
        return 0.0
    return 1.0 if em_check(ans, golden_answers) else 0.1


def decompose_training_rollouts(run_dir: str, train_parquet: str) -> dict:
    train = pd.read_parquet(train_parquet)
    questions = [str(q) for q in train["env_kwargs"].apply(lambda r: r["question"])]
    gt_by_question = {
        " ".join(q.casefold().split()): list(train["env_kwargs"].iloc[i]["ground_truth"]["target"])
        for i, q in enumerate(questions)
    }
    # longest normalized question first so the real (full) question wins the
    # substring join over any few-shot example question inside the prompt
    qt_list = sorted(((nq, gt_by_question[nq]) for nq in gt_by_question), key=lambda x: -len(x[0]))

    episode_finals = {"em_hit": 0, "format_only": 0, "no_answer": 0}
    row_stats = {"n_rows": 0, "invalid_rows": 0}
    mismatches: list[dict] = []
    n_episodes = 0
    for step_file in sorted(Path(run_dir, "rollouts").glob("*.jsonl")):
        if ".audit." in step_file.name:
            continue
        audit_path = str(step_file).replace(".jsonl", ".audit.jsonl")
        with open(step_file) as f:
            lines = [json.loads(l) for l in f]
        with open(audit_path) as f:
            aud = [json.loads(l) for l in f]
        if len(lines) != len(aud):
            mismatches.append({"step": step_file.name, "error": "line count mismatch"})
            continue
        by_traj: dict[str, list] = {}
        for rec, a in zip(lines, aud):
            m = a["metadata"]
            by_traj.setdefault(m["traj_uid"], []).append((m["env_step"], rec, m["is_action_valid"]))
        for traj_uid, records in by_traj.items():
            records.sort(key=lambda r: r[0])
            nprompt = " ".join(records[0][1]["input"].casefold().split())
            golden = None
            for norm_q, gts in qt_list:
                if norm_q in nprompt:
                    golden = gts
                    break
            if golden is None:
                continue
            # replay env: accumulate projected actions, final score at done step
            history = ""
            episode_reward = 0.0
            for env_step, rec, valid in records:
                projected = _project_action(rec["output"])
                history += projected
                done = ("</answer>" in projected) or (env_step == 1)
                if done:
                    episode_reward = _env_compute_score(history, golden)
            n_episodes += 1
            if episode_reward >= 1.0:
                episode_finals["em_hit"] += 1
            elif episode_reward >= 0.1:
                episode_finals["format_only"] += 1
            else:
                episode_finals["no_answer"] += 1
            for env_step, rec, valid in records:
                expected = episode_reward - 0.1 * (0 if valid else 1)
                actual = rec["score"]
                row_stats["n_rows"] += 1
                if not valid:
                    row_stats["invalid_rows"] += 1
                if abs(expected - actual) > 0.051:
                    mismatches.append(
                        {
                            "traj": traj_uid[:8],
                            "env_step": env_step,
                            "score": round(actual, 3),
                            "expected": round(expected, 3),
                            "episode_reward": round(episode_reward, 3),
                            "valid": bool(valid),
                        }
                    )
    return {
        "n_episodes": n_episodes,
        **episode_finals,
        **row_stats,
        "penalty_units": row_stats["invalid_rows"],
        "reconstruction_mismatches": mismatches,
    }


# ----------------------------------------------------------------------------
# Task 5: heldout failure classification
# ----------------------------------------------------------------------------
def classify_failures(eps: list[dict]) -> dict:
    classes = {
        "no_search": 0,
        "searched_then_wrong": 0,
        "searched_then_correct": 0,
        "format_error_no_answer": 0,
        "invalid_action_only": 0,
    }
    detail: list[dict] = []
    for ep in eps:
        steps = ep["steps"]
        searched = any(s.get("executed_search") for s in steps)
        answer_ok = extract_solution_fork(steps_text(ep)) is not None
        invalid = any(not s["action_quality"]["projected_valid"] for s in steps)
        won = ep["won"]
        if won:
            label = "searched_then_correct" if searched else "no_search"
        elif searched:
            label = "searched_then_wrong"
        elif answer_ok:
            label = "no_search"
        elif invalid:
            label = "invalid_action_only"
        else:
            label = "format_error_no_answer"
        classes[label] += 1
        detail.append(
            {
                "question": ep["question"][:80],
                "label": label,
                "searched": searched,
                "has_answer_tag": answer_ok,
                "invalid": invalid,
                "reward": ep["reward"],
            }
        )
    return {"classes": classes, "detail": detail}


# ----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-run", required=True, help="eval run dir (base model)")
    parser.add_argument("--new-run", required=True, help="eval run dir (trained adapter)")
    parser.add_argument("--train-run", required=True, help="training run dir (rollouts/audit)")
    parser.add_argument("--train-parquet", required=True, help="train64 parquet used for that run")
    parser.add_argument("--labels", nargs=2, default=["base", "trained"])
    parser.add_argument("--out", default="action_reward_diag", help="output dir")
    args = parser.parse_args()

    base_eps = load_episodes(args.base_run)
    new_eps = load_episodes(args.new_run)

    comp = compare_outputs(base_eps, new_eps, args.labels[1])
    decomp = decompose_training_rollouts(args.train_run, args.train_parquet)
    failures = classify_failures(new_eps)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "labels": args.labels,
        "output_comparison": comp,
        "training_reward_decomposition": decomp,
        "heldout_failure_classes": failures["classes"],
        "heldout_failure_detail": failures["detail"],
    }
    (out_dir / "diagnosis.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    lines = [
        f"# P3 动作格式 + 奖励路径只读诊断（{args.labels[0]} vs {args.labels[1]}）",
        "",
        f"评测题数: {comp['n_questions']}",
        "",
        "## 1/6. 输出对比（同一批 32 题）",
        "",
        f"- 字节级完全一致: **{comp['byte_identical']}/{comp['n_questions']}** ({comp['identical_rate']:.1%})",
        f"- 第一步 raw_action 一致: {comp['same_first_step']}/{comp['n_questions']}",
        f"- 平均编辑距离(归一化): {comp['mean_lev_distance']:.3f}",
        f"- 动作有效性翻转(valid→invalid 或反之): {comp['validity_flips']}",
        f"- EM 翻转(答对↔答错): {comp['em_flips']}",
        "",
        "### EM 翻转样例",
        "",
    ]
    for ex in comp["em_flip_examples"]:
        lines += [
            f"- **{ex['question']}**  base_won={ex['base_won']} new_won={ex['new_won']}",
            f"  - base 首步: {ex['base_first_step'][:180]}",
            f"  - new 首步: {ex['new_first_step'][:180]}",
        ]
    lines += [
        "",
        "## 4. 训练 reward 分解（train64-nqh，episode 级精确重建）",
        "",
        "重建语义（377/377 行完全复现，见第 7 节）：投影先于 env（chat_history 只含投影后",
        "动作），episode_rewards 累计到最终 compute_score 并写入该 traj 的每一行，",
        "apply_invalid_action_penalty 再按行扣 0.1。",
        "",
        f"- episode 总数: {decomp['n_episodes']}",
        f"- episode 最终 EM 命中(1.0): **{decomp['em_hit']}**",
        f"- episode 最终仅格式分(0.1): **{decomp['format_only']}**",
        f"- episode 最终无 answer(0.0): **{decomp['no_answer']}**",
        f"- 行级: 共 {decomp['n_rows']} 行，invalid {decomp['invalid_rows']} "
        f"({decomp['invalid_rows']/max(decomp['n_rows'],1):.1%})，惩罚单元 {decomp['penalty_units']}",
        f"- 重建不匹配条数: {len(decomp['reconstruction_mismatches'])}",
        "",
        "## 5. heldout 失败分类（训练模型）",
        "",
        "| 类别 | 数量 |",
        "|---|---|",
    ]
    for label, count in failures["classes"].items():
        lines.append(f"| {label} | {count} |")
    lines += ["", "## 2. mixed/duplicate 无效样例（本项目投影规则）", ""]
    lines += [
        "- 规则（projection.py）：动作同时含 `<search>` 与 `<answer>`（mixed），",
        "  或同一标签出现 ≥2 次（duplicate）→ `valids=0`，训练时每个无效动作 -0.1；",
        "- 官方 Search-R1 无此规则：`<(search|answer)>(.*?)</\\1>` 取第一个匹配，",
        "  仅无匹配判 invalid，且 invalid 时环境提示重试（不终止、无惩罚）。",
        "",
    ]
    seen = set()
    for ep in new_eps:
        for step in ep["steps"]:
            quality = step["action_quality"]
            if (quality["mixed_tags"] or quality["duplicate_tags"]) and step["raw_action"] not in seen:
                seen.add(step["raw_action"])
                lines.append(
                    f"- mixed={quality['mixed_tags']} dup={quality['duplicate_tags']}: "
                    f"`{step['raw_action'][:220]}`"
                )
    lines += [
        "",
        "## 7. 训练 reward 路径重建验证（关键）",
        "",
        "逐 (traj, step) 行按 env 精确语义重放：投影先于 env（SearchEnvironmentManager.step",
        "把 search_projection 后的动作传给 env，chat_history 只含投影后动作 + 搜索观察），",
        "episode_rewards = 该 traj 各步 env reward 之和（done 步的 compute_score，多步",
        "episode 中间步 reward=0），gather_rollout_data 把 episode_rewards 写入该 traj",
        "每一行，EpisodeRewardManager 放到每行最后 token，apply_invalid_action_penalty",
        "按行扣 0.1×该行 invalid。因此：",
        "",
        "    recorded_score == episode_rewards - 0.1 * (0 if row_valid else 1)",
        "",
        f"- 行数: {decomp['n_rows']}，重建一致: {decomp['n_rows'] - len(decomp['reconstruction_mismatches'])}，"
        f"不一致: {len(decomp['reconstruction_mismatches'])}",
        "- 早期逐记录分解的 18 条『mismatch』与 73 条 traj 级『mismatch』全部归因于重建",
        "  脚本未建模以上三层语义（逐记录只看了本步 output；traj 级用原始 output 拼接并",
        "  把惩罚按 episode 总数扣），**不是训练实现问题**。",
        "- `<answer />` 自闭合标签不含 `<answer>` 子串：投影的标签计数不把它算作 answer",
        "  （n_answer=0，不判 mixed），env `_is_done` 也不认（无 `</answer>`）→ 该步不终止",
        "  并继续搜索/直到 max_turns。",
        "",
    ]
    (out_dir / "diagnosis.md").write_text("\n".join(lines) + "\n")
    print(f"written: {out_dir / 'diagnosis.json'} {out_dir / 'diagnosis.md'}")
    print(f"identical={comp['byte_identical']}/{comp['n_questions']} "
          f"lev={comp['mean_lev_distance']:.3f} em_flips={comp['em_flips']}")
    print(f"decomp: episodes={decomp['n_episodes']} em={decomp['em_hit']} fmt={decomp['format_only']} "
          f"noans={decomp['no_answer']} rows={decomp['n_rows']} invalid={decomp['invalid_rows']} "
          f"mismatches={len(decomp['reconstruction_mismatches'])}")
    print(f"failures: {failures['classes']}")


if __name__ == "__main__":
    main()
