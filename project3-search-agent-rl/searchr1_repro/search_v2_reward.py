"""Search-aware GRPO clean v2 step-reward components (single implementation source).

Frozen formula (user directive 2026-08-22, docs/P3_SEARCH_AWARE_GRPO10_RESULT_REPORT_2026-08-22.md):

    R = R_answer
        + 0.15 * first_evidence_hit
        + 0.30 * searched_and_correct_and_evidence_hit
        - 0.20 * invalid_or_error
        - 0.20 * true_redundant_search
        - 0.20 * new_answer_leak_in_query

Frozen semantics (v2, clean-upstream baseline):
- format_score = 0.0 (identical to the clean baseline; NO v1 0.1 format reward)
- valid_retrieval itself earns nothing (alpha = 0)
- first_evidence_hit: at most ONE evidence credit per trajectory (anti
  search-spam; later evidence-hitting searches earn 0 credit, no penalty)
- searched_and_correct_and_evidence_hit (sce): settled once per trajectory on
  the terminal step iff EM AND at least one real successful evidence-hitting
  (non-leak) retrieval happened
- TRUE redundancy (v2 fix; the v1 rule penalized every 2nd+ search):
  a search step is redundant iff it is NOT the first search AND any of:
      (a) its normalized query equals a query already issued in this episode, or
      (b) the Retriever result brings NO document ID not already seen, or
      (c) document IDs are unstable/absent AND the normalized document-body
          hash equals one already seen (no new document by content)
  The following are NEVER redundant: a different query that returns at least
  one new document; a second search that brings new evidence in a multi-hop
  task; the first (valid) search. Invalid/error steps pay the invalid penalty
  only (never redundant).
- invalid_or_error: empty/None query or retrieval status not in
  {success, no_results} (api_error / processing_error / invalid_query /
  tool_exception / unknown states) -- includes error observations
- answer leak: token-boundary matcher, question-exclusion rule (an alias
  already present in the question is never a leak), MIN_ALIAS_LEN filter
- evidence_hit checks ONLY the real Retriever-returned document bodies (never
  query text, error text, or model output)
- Observation tokens never enter the policy loss (mask unchanged)
- trajectory return grouped by traj_uid; GRPO mean/std over the 5 trajectory
  returns of a uid; trajectory advantage broadcast to every record of the
  trajectory; per-trajectory component sum == placed sum == trajectory return
  (fail-closed, exact integer cents)

Step attribution:
- R_answer + sce on the terminal answer step only; evidence/invalid/redundant/
  leak shaping on the corresponding search step

All arithmetic is integer cents so component sums are exact (no binary float
drift); use reward_float() at the boundary.

This module is the SINGLE implementation source shared by:
- the training-side env computation (patch v2-0004, vendored SearchEnv)
- the offline historical-rollout replay (scripts/p3_v2_reward_replay.py)
- the CPU unit tests (tests/test_search_v2_reward.py)
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Any

# ---------------------------------------------------------------------------
# Frozen coefficients (cents)
# ---------------------------------------------------------------------------
FORMAT_SCORE_C = 0           # format_score = 0.0 (clean baseline, no v1 0.1)
ANSWER_REWARD_C = 100        # EM correct -> R_answer = 1.0
EVIDENCE_HIT_C = 15          # 0.15, at most once per trajectory
SCE_C = 30                   # 0.30 searched-and-correct-and-evidence (once)
INVALID_C = -20              # -0.20 invalid/error search step
REDUNDANT_C = -20            # -0.20 per TRUE redundant search step
ANSWER_LEAK_C = -20          # -0.20 answer-leak search step

MIN_ALIAS_LEN = 2            # normalized aliases shorter than this are excluded
SUCCESS_STATUSES = {"success", "no_results"}  # env retrieval_failed semantics


def norm_text(s: str) -> str:
    """NFKC + casefold + fold whitespace/punctuation (shared convention)."""
    s = unicodedata.normalize("NFKC", str(s))
    s = s.casefold()
    s = re.sub(r"[\s\W_]+", "", s)
    return s


def norm_query(query: str | None) -> str:
    """Normalized query string for duplicate-query detection.

    NFKC + casefold + whitespace collapse (no punctuation stripping: the query
    text itself is the identity key, not a searchable alias).
    """
    if not query:
        return ""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(query)).casefold()).strip()


def _tokens(text: str | None) -> list[str]:
    """Unicode-aware tokenization for alias matching (NFKC + casefold + \\w+ runs)."""
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
    return " ".join(words)


def valid_aliases(gt_targets: list[str]) -> list[list[str]]:
    """Ground-truth aliases as token sequences; too-short/empty ones are excluded."""
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

    `doc_text` must be the REAL document text returned by the Retriever.
    """
    if not doc_text or not gt_aliases:
        return False
    doc_tokens = _tokens(doc_text)
    return any(_phrase_in(a, doc_tokens) for a in gt_aliases)


def answer_leak_in_query(
    query: str | None, gt_aliases: list[list[str]], question: str | None
) -> dict[str, Any]:
    """Answer-leak check for one search query (token-boundary phrase matching)."""
    if not query or not gt_aliases:
        return {"leak": False, "alias": None}
    query_tokens = _tokens(query)
    question_tokens = _tokens(question) if question else []
    for a in gt_aliases:
        if _phrase_in(a, query_tokens) and not _phrase_in(a, question_tokens):
            return {"leak": True, "alias": alias_str(a)}
    return {"leak": False, "alias": None}


def doc_ids_key(document_ids: list[str] | None) -> str | None:
    """Stable document-ID signature; None when no stable IDs are available."""
    if document_ids is None:
        return None
    return "\x1f".join(str(d) for d in document_ids)


def content_hash(doc_text: str | None) -> str:
    """Normalized document-body hash used when document IDs are unstable/absent."""
    return hashlib.sha256(norm_text(doc_text or "").encode()).hexdigest()


def is_true_redundant(
    *,
    query: str | None,
    status: str | None,
    doc_ids: list[str] | None,
    doc_text: str | None,
    prior_queries: set[str],
    prior_doc_ids: set[str],
    prior_content_hashes: set[str],
    is_first_search: bool,
) -> bool:
    """v2 true-redundancy verdict (see module docstring for the definition).

    Applies ONLY to valid (non-invalid) search steps; invalid steps pay the
    invalid penalty and are never additionally redundant.
    """
    if is_first_search:
        return False
    nq = norm_query(query)
    if nq and nq in prior_queries:
        return True
    if doc_ids is not None:
        # document IDs available: redundant iff NO new id appears
        return not any(d not in prior_doc_ids for d in doc_ids)
    # no stable document IDs -> normalized content hash comparison
    h = content_hash(doc_text)
    return h in prior_content_hashes


def search_step_components_v2(
    *,
    query: str | None,
    status: str | None,
    doc_ids: list[str] | None,
    doc_text: str | None,
    gt_aliases: list[list[str]],
    question: str | None,
    prior_queries: set[str],
    prior_doc_ids: set[str],
    prior_content_hashes: set[str],
    is_first_search: bool,
    had_evidence_credit: bool,
) -> dict[str, Any]:
    """Per-search-step v2 components (all values in cents), plus state updates.

    Returns a single dict whose "state" key carries the updated tracker sets
    (prior_queries / prior_doc_ids / prior_content_hashes / had_evidence_credit)
    which the caller MUST apply to the episode state (the env keeps it in
    self.*; the replay keeps it in a local dict).
    """
    invalid = bool(not query or not str(query).strip() or status not in SUCCESS_STATUSES)
    redundant = bool(not invalid and is_true_redundant(
        query=query, status=status, doc_ids=doc_ids, doc_text=doc_text,
        prior_queries=prior_queries, prior_doc_ids=prior_doc_ids,
        prior_content_hashes=prior_content_hashes, is_first_search=is_first_search,
    ))
    evidence_raw = evidence_hit_in_docs(doc_text, gt_aliases)
    leak = answer_leak_in_query(query, gt_aliases, question)
    evidence_effective = bool(evidence_raw and not invalid and not leak["leak"])
    # v2: at most ONE evidence credit per trajectory (anti search-spam); a
    # non-redundant later evidence-hitting search earns 0 credit, no penalty.
    evidence_credit = bool(evidence_effective and not redundant and not had_evidence_credit)

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
    components["version"] = "v2"

    # state updates (track everything, including invalid steps' queries so a
    # later identical valid query still counts as a duplicate)
    nq = norm_query(query)
    new_prior_queries = set(prior_queries)
    if nq:
        new_prior_queries.add(nq)
    new_prior_doc_ids = set(prior_doc_ids)
    if doc_ids is not None:
        new_prior_doc_ids.update(str(d) for d in doc_ids)
    new_prior_content_hashes = set(prior_content_hashes)
    if doc_ids is None and doc_text:
        new_prior_content_hashes.add(content_hash(doc_text))
    components["state"] = {
        "prior_queries": new_prior_queries,
        "prior_doc_ids": new_prior_doc_ids,
        "prior_content_hashes": new_prior_content_hashes,
        "had_evidence_credit": had_evidence_credit or evidence_credit,
    }
    return components


def terminal_step_components_v2(
    *,
    r_answer_total: float,
    em: bool,
    had_effective_evidence: bool,
) -> dict[str, Any]:
    """Terminal (answer) step components.

    r_answer_total is the clean env score (compute_score, format_score=0.0) in
    {0.0, 1.0}; format reward is always 0 (clean baseline). sce applies once
    iff EM AND the episode had at least one real successful evidence-hitting
    (non-leak) search.
    """
    r_answer_c = ANSWER_REWARD_C if em else 0
    format_c = FORMAT_SCORE_C  # 0.0, clean baseline (v1's 0.1 is gone)
    sce_c = SCE_C if (em and had_effective_evidence) else 0
    return {
        "terminal": True,
        "r_answer_total": r_answer_total,
        "answer_reward_c": r_answer_c,
        "format_reward_c": format_c,
        "sce_c": sce_c,
        "had_effective_evidence": bool(had_effective_evidence),
        "version": "v2",
    }


def episode_totals(step_comps: list[dict[str, Any]], terminal_comp: dict[str, Any] | None) -> dict[str, Any]:
    """Aggregate one episode's 8 recorded reward components (in cents).

    The 8 components (same schema as v1 for audit-pipeline compatibility):
    answer_reward / format_reward / evidence_hit_reward / searched_correct_bonus /
    invalid_penalty / redundant_penalty / answer_leak_penalty / total_reward.
    The component sum equals total_reward exactly (integer cents).
    """
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
        "version": "v2",
    }


def reward_float(cents: int) -> float:
    """Exact cents -> float reward value."""
    return cents / 100.0
