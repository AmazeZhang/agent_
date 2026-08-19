"""Search-aware GRPO v1 step-reward components (single implementation source).

Frozen formula (Phase 4B, 2026-08-19, docs/P3_PHASE4_SEARCH_CAUSAL_DIAG_RESULT_2026-08-19.md §9):

    R = R_answer + 0.15*evidence_hit + 0.30*searched_and_correct_and_evidence_hit
        - 0.20*invalid_or_error - 0.45*redundant_search_count - 0.20*new_answer_leak_in_query

Frozen semantics:
- format_score = 0.1 (this round does not change format reward)
- valid_retrieval coefficient alpha = 0: an irrelevant search itself earns nothing
- evidence_hit checks ONLY the real document bodies returned by the Retriever
  (never the query text, error text, or model output)
- searched_and_correct_and_evidence_hit (sce) requires at least one real
  successful retrieval whose returned evidence hit AND the final answer is EM-correct
- answer-leak rule: a ground-truth alias occurs in the search query as a full
  token phrase and that same phrase was NOT already present in the question;
  too-short/empty aliases are excluded (MIN_ALIAS_LEN); a leaking step
  contributes neither evidence_hit nor sce (its evidence reward is zeroed) and
  additionally pays -0.20

Token-boundary alias matching (Phase 4B.1, 2026-08-13): aliases are normalized
to Unicode token sequences (NFKC + casefold + \\w+ runs). A multi-word alias
("the City of Lights") matches only as a CONTIGUOUS token subsequence of the
text; a short alias ("us", "it") matches only as a standalone token, never
inside a longer word ("museum", "britain"). Punctuation/whitespace differences
between alias and text are folded away by tokenization.

Step attribution:
- R_answer only on the terminal answer step; evidence/invalid/leak/redundant
  shaping on the corresponding search step; sce settled on the terminal step via
  episode metadata (had_effective_evidence) linked to real successful retrievals

All arithmetic is integer cents so component sums are exact (no binary float
drift); use reward_float() at the boundary.

This module is the SINGLE implementation source shared by:
- the training-side env computation (patch 0007, vendored SearchEnv)
- the offline historical-rollout replay (scripts/p3_v1_reward_replay.py)
- the CPU unit tests (tests/test_search_v1_reward.py)
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

# ---------------------------------------------------------------------------
# Frozen coefficients (cents)
# ---------------------------------------------------------------------------
FORMAT_SCORE_C = 10          # format_score = 0.1
ANSWER_REWARD_C = 100        # EM correct -> R_answer = 1.0
EVIDENCE_HIT_C = 15          # 0.15 per evidence-hitting search step
SCE_C = 30                   # 0.30 searched-and-correct-and-evidence bonus
INVALID_C = -20              # -0.20 invalid/error search step
REDUNDANT_C = -45            # -0.45 per redundant (2nd+) search step
ANSWER_LEAK_C = -20          # -0.20 answer-leak search step

MIN_ALIAS_LEN = 2            # normalized aliases shorter than this are excluded
SUCCESS_STATUSES = {"success", "no_results"}  # env retrieval_failed semantics


def norm_text(s: str) -> str:
    """NFKC + casefold + fold whitespace/punctuation (diag1/diag4 convention).

    Retained for backward compatibility with the diag pipelines; the alias
    matcher itself uses token sequences (see _tokens / valid_aliases).
    """
    s = unicodedata.normalize("NFKC", str(s))
    s = s.casefold()
    s = re.sub(r"[\s\W_]+", "", s)
    return s


def _tokens(text: str | None) -> list[str]:
    """Unicode-aware tokenization for alias matching (NFKC + casefold + \\w+ runs).

    Whitespace and punctuation separate tokens, so a two-character alias ("us",
    "it") can only ever match the standalone token "us"/"it" -- never a
    substring inside "museum" or "britain". Used for BOTH the alias side (a
    multi-word alias becomes its token sequence) and the searched text side.
    """
    if not text:
        return []
    return re.findall(r"\w+", unicodedata.normalize("NFKC", str(text)).casefold())


def _phrase_in(words: list[str], text_tokens: list[str]) -> bool:
    """Contiguous token-subsequence match: the alias token sequence appears in the text."""
    n = len(words)
    if n == 0 or len(text_tokens) < n:
        return False
    for i in range(len(text_tokens) - n + 1):
        if text_tokens[i : i + n] == words:
            return True
    return False


def alias_str(words: list[str]) -> str:
    """Human-readable alias phrase (audit field / test assertions), e.g.
    ["city", "of", "lights"] -> "city of lights"."""
    return " ".join(words)


def valid_aliases(gt_targets: list[str]) -> list[list[str]]:
    """Ground-truth aliases as token sequences; too-short/empty ones are excluded.

    An alias must tokenize to >= 1 word with total normalized length >=
    MIN_ALIAS_LEN ("P" and "" drop out; "pa", "us", "it", "paris" survive).
    Returned aliases only ever match text as full token phrases (see _tokens).
    """
    out: list[list[str]] = []
    for t in gt_targets:
        if not t:
            continue
        words = _tokens(t)
        if not words:
            continue
        if sum(len(w) for w in words) < MIN_ALIAS_LEN:
            continue
        out.append(words)
    return out


def evidence_hit_in_docs(doc_text: str | None, gt_aliases: list[list[str]]) -> bool:
    """Evidence hit: a valid gt alias occurs as a full token phrase in the docs.

    `doc_text` must be the REAL document text returned by the Retriever (the
    env passes the tool output; the replay passes corpus text by document_ids).
    Token-boundary matching: "us" never hits "museum", "it" never hits
    "britain"; "US" hits the standalone token "us".
    """
    if not doc_text or not gt_aliases:
        return False
    doc_tokens = _tokens(doc_text)
    return any(_phrase_in(a, doc_tokens) for a in gt_aliases)


def answer_leak_in_query(
    query: str | None, gt_aliases: list[list[str]], question: str | None
) -> dict[str, Any]:
    """Answer-leak check for one search query (token-boundary phrase matching).

    Returns {"leak": bool, "alias": str|None}:
      leak == True iff a valid gt alias phrase occurs in the query tokens AND
      that same phrase does NOT occur in the question tokens (a question that
      itself contains the answer alias must never be misjudged). "alias" is the
      matched phrase (casefolded, space-joined), e.g. "paris" or "city of lights".
    """
    if not query or not gt_aliases:
        return {"leak": False, "alias": None}
    query_tokens = _tokens(query)
    question_tokens = _tokens(question) if question else []
    for a in gt_aliases:
        if _phrase_in(a, query_tokens) and not _phrase_in(a, question_tokens):
            return {"leak": True, "alias": alias_str(a)}
    return {"leak": False, "alias": None}


def search_step_components(
    *,
    query: str | None,
    status: str | None,
    doc_text: str | None,
    gt_aliases: list[list[str]],
    question: str | None,
    prior_search_count: int,
) -> dict[str, Any]:
    """Per-search-step v1 components (all values in cents).

    Args:
        query: the executed search query (None/"" -> invalid)
        status: retrieval status from the env ("success"/"no_results"/"invalid_query"/...)
        doc_text: real document bodies returned by the Retriever (tool output)
        gt_aliases: valid_aliases(gt targets) -- token-sequence aliases
        question: the original question text (for the answer-leak rule)
        prior_search_count: number of search steps BEFORE this one in the episode
    """
    invalid = bool(not query or not str(query).strip() or status not in SUCCESS_STATUSES)
    evidence_raw = evidence_hit_in_docs(doc_text, gt_aliases)
    leak = answer_leak_in_query(query, gt_aliases, question)
    evidence_effective = bool(evidence_raw and not invalid and not leak["leak"])
    redundant = bool(prior_search_count >= 1)
    # Redundant (2nd+) search steps earn NO evidence credit: the -0.45 penalty
    # alone covers the step. Without this, a 2-evidence-search episode would
    # score 1.15 > 1.0, violating the hard gate "redundant-search-correct <=
    # direct-correct" (diag T6: 2 useful searches == 1.00 == T1). evidence_effective
    # stays True for sce linkage: sce is settled via episode metadata and the
    # sce rule only requires one real successful evidence-hitting retrieval.
    evidence_credit = bool(evidence_effective and not redundant)

    components = {
        "evidence_hit": evidence_raw,
        "evidence_effective": evidence_effective,
        "evidence_credit": evidence_credit,
        "invalid_or_error": invalid,
        "redundant_search": redundant,
        "answer_leak": leak["leak"],
        "answer_leak_alias": leak["alias"],
    }
    step_shaping_c = 0
    if evidence_credit:
        step_shaping_c += EVIDENCE_HIT_C
    if invalid:
        step_shaping_c += INVALID_C
    if redundant:
        step_shaping_c += REDUNDANT_C
    if leak["leak"]:
        step_shaping_c += ANSWER_LEAK_C
    components["step_shaping_c"] = step_shaping_c
    components["query"] = query
    components["status"] = status
    return components


def terminal_step_components(
    *,
    r_answer_total: float,
    em: bool,
    had_effective_evidence: bool,
) -> dict[str, Any]:
    """Terminal (answer) step components.

    r_answer_total is the env score (compute_score) in {0.0, 0.1, 1.0};
    decomposed into answer_reward (1.0 iff EM) and format_reward (0.1 iff an
    <answer> was produced but not EM). sce bonus applies iff EM AND the episode
    had at least one real successful evidence-hitting (non-leak) search.
    """
    r_answer_c = ANSWER_REWARD_C if em else 0
    format_c = FORMAT_SCORE_C if (not em and r_answer_total > 0.0) else 0
    sce_c = SCE_C if (em and had_effective_evidence) else 0
    return {
        "terminal": True,
        "r_answer_total": r_answer_total,
        "answer_reward_c": r_answer_c,
        "format_reward_c": format_c,
        "sce_c": sce_c,
        "had_effective_evidence": bool(had_effective_evidence),
    }


def episode_totals(step_comps: list[dict[str, Any]], terminal_comp: dict[str, Any] | None) -> dict[str, Any]:
    """Aggregate one episode's 8 recorded reward components (in cents).

    The 8 components (per Phase 4B requirement): answer_reward / format_reward /
    evidence_hit_reward / searched_correct_bonus / invalid_penalty /
    redundant_penalty / answer_leak_penalty / total_reward. The component sum is
    asserted to equal total_reward (exact integer arithmetic).
    """
    # exactly EVIDENCE_HIT_C per credit-bearing step (evidence_credit excludes
    # redundant steps) -- NEVER c["step_shaping_c"], which would double-count the
    # invalid/redundant/leak penalties of the same step. Mirrors the reward
    # manager's per-step accounting in episode.py.
    evidence_c = sum(EVIDENCE_HIT_C for c in step_comps if c.get("evidence_credit"))
    invalid_c = sum(INVALID_C for c in step_comps if c.get("invalid_or_error"))
    redundant_c = sum(REDUNDANT_C for c in step_comps if c.get("redundant_search"))
    leak_c = sum(ANSWER_LEAK_C for c in step_comps if c.get("answer_leak"))
    answer_c = terminal_comp.get("answer_reward_c", 0) if terminal_comp else 0
    format_c = terminal_comp.get("format_reward_c", 0) if terminal_comp else 0
    sce_c = terminal_comp.get("sce_c", 0) if terminal_comp else 0

    total_c = answer_c + format_c + evidence_c + sce_c + invalid_c + redundant_c + leak_c
    return {
        "answer_reward_c": answer_c,
        "format_reward_c": format_c,
        "evidence_hit_reward_c": evidence_c,
        "searched_correct_bonus_c": sce_c,
        "invalid_penalty_c": invalid_c,
        "redundant_penalty_c": redundant_c,
        "answer_leak_penalty_c": leak_c,
        "total_reward_c": total_c,
        "n_search_steps": len(step_comps),
    }


def reward_float(cents: int) -> float:
    """Exact cents -> float reward value."""
    return cents / 100.0
