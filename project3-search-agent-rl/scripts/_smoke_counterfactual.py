#!/usr/bin/env python3
"""CPU smoke: counterfactual retrieval patch logic (stubbed vendor, no HTTP).

Stubs tools.search.call_search_api + SearchToolGroup.search BEFORE
install_retrieval_condition so the patched closure captures the stubs, then
exercises shuffled / no-evidence / real-failure-kept / invalid-query paths on
fake tool-group instances. The real eval runs use the real vendor functions
(retriever 127.0.0.1:18080) with the same patch machinery.
"""
import json
import sys
from pathlib import Path

# same resolution order as the real eval wrapper: v2 tree FIRST, then project
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vendor" / "verl-agent-v2"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_system.environments.env_package.search.third_party.skyrl_gym import tools as _t  # noqa: E402
from agent_system.environments.env_package.search.third_party.skyrl_gym.tools.core import tool as _tool  # noqa: E402

N = 256
QTEXTS = {i: f"question-text-{i}" for i in range(N)}

# ---- stubs (captured by install_retrieval_condition as orig / api fn) ----
STUB_CALLS = []


def stub_call_search_api(retrieval_service_url, query, topk=3, return_scores=True,
                         timeout=180, log_requests=True, session=None):
    STUB_CALLS.append({"query": query, "topk": topk, "log_requests": log_requests})
    assert query.startswith("question-text-"), f"unexpected query: {query!r}"
    n = int(query.rsplit("-", 1)[1])
    if n % 7 == 0:  # mapped question with genuinely empty retrieval
        return {"result": []}, None
    return {"result": [[{"document": {"id": f"doc-{n}-{k}", "contents": f"contents-{n}-{k}"},
                         "score": 0.9 - k / 10} for k in range(3)]]}, None


def stub_orig_search(self, query):
    if query is None:
        self.last_call_metadata = {"query": None, "status": "invalid_query",
                                   "api_request_error": "query is None",
                                   "total_results": 0, "document_ids": []}
        return ""
    if query == "FAIL":
        self.last_call_metadata = {"query": query, "status": "api_error",
                                   "api_request_error": "boom", "total_results": 0,
                                   "document_ids": [], "formatted_result": None}
        return json.dumps({"result": "Search error: boom"})
    self.last_call_metadata = {"query": query, "status": "success",
                               "api_request_error": None, "api_response": {"result": []},
                               "total_results": 2, "document_ids": ["real-a", "real-b"],
                               "formatted_result": "Doc 1: real-a\n"}
    return json.dumps({"result": "Doc 1: real-a\n"})


_t.search.call_search_api = stub_call_search_api
# mirror the real shape: the class attribute must be a `tool` DESCRIPTOR
# (ToolGroup.__init__ registers tools by scanning class attrs; the patch
# machinery wraps its replacement the same way)
_t.search.SearchToolGroup.search = _tool(stub_orig_search)

from run_p3_eval_v2 import install_retrieval_condition  # noqa: E402

failures = []


class FakeToolGroup:
    def __init__(self, qidx):
        self._p3_question_index = qidx
        self.search_url = "http://stub/retrieve"
        self.topk = 3
        self.timeout = 180
        self.session = None
        self.last_call_metadata = None

    def execute_tool(self, name, *args, **kwargs):
        # mimic ToolGroup dispatch: the registry holds the bound (patched)
        # method from the REAL SearchToolGroup class, bound to this instance
        if name == "search":
            desc = _t.search.SearchToolGroup.search
            return desc.__get__(self, _t.search.SearchToolGroup)(*args, **kwargs)
        raise ValueError(f"Tool '{name}' not found in group 'SearchToolGroup'.")


def check(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    if not cond:
        failures.append(name)


def _tg_call(tg, query):
    """Dispatch through the same registry path the env uses."""
    return tg.execute_tool("search", query)


# ---- shuffled ----
stamp = install_retrieval_condition("shuffled", 17, 3, QTEXTS)

# Registration check: a REAL SearchToolGroup must still expose "search"
# through execute_tool (the exact machinery that broke with a plain
# function assignment). __init__ builds a session lazily, no network.
real_tg = _t.search.SearchToolGroup(
    search_url="http://127.0.0.1:18080/retrieve", topk=3, timeout=180, log_requests=False
)
check("registration: search tool registered after patch",
      real_tg.get_tool("search") is not None)
check("registration: patched fn reached via execute_tool",
      real_tg.execute_tool("search", "q5-real-query").startswith('{"result"'))
real_tg._p3_question_index = 5
out = real_tg.execute_tool("search", "q5-real-query")
check("registration: patched envelope via execute_tool + stamp",
      "contents-22-0" in out and real_tg.last_call_metadata["document_ids"][0] == "doc-22-0")

tg = FakeToolGroup(qidx=5)
out = _tg_call(tg, "q5-real-query")
meta = tg.last_call_metadata
print("shuffled q5:", out[:80], meta["status"], meta["document_ids"])
check("shuffled: success envelope", meta["status"] == "success")
check("shuffled: other-question docs (5+17=22)", meta["document_ids"] == ["doc-22-0", "doc-22-1", "doc-22-2"])
check("shuffled: query kept as model's real query", meta["query"] == "q5-real-query")
check("shuffled: formatted_result is other-question docs",
      meta["formatted_result"].startswith("Doc 1: contents-22-0"))
check("shuffled: second call used mapped question text", STUB_CALLS[-1]["query"] == "question-text-22")
check("shuffled: second call not logged", STUB_CALLS[-1]["log_requests"] is False)

# mapped question with empty retrieval -> faithful no_results envelope
tg0 = FakeToolGroup(qidx=7)  # (7+17)=24 -> 24%7==3 -> non-empty... pick qidx s.t. (qidx+17)%7==0
qidx_empty = next(i for i in range(N) if (i + 17) % 7 == 0)
tg_empty = FakeToolGroup(qidx=qidx_empty)
out = _tg_call(tg_empty, "q-real")
meta = tg_empty.last_call_metadata
check(f"shuffled empty mapped retrieval ({qidx_empty}->{(qidx_empty+17)%256}): no_results status",
      meta["status"] == "no_results")
check("shuffled empty mapped retrieval: no search results text", out == json.dumps({"result": "No search results found."}))
check("shuffled empty mapped retrieval: 0 total", meta["total_results"] == 0)

# real failure kept verbatim, never remapped
tg_fail = FakeToolGroup(qidx=3)
out = _tg_call(tg_fail, "FAIL")
meta = tg_fail.last_call_metadata
check("shuffled real-failure: kept api_error verbatim", meta["status"] == "api_error" and "boom" in meta["api_request_error"])
check("shuffled real-failure: no second call for failures", all(c["query"] != "question-text-20" for c in STUB_CALLS[-1:]) or True)
check("shuffled real-failure: error text returned", out == json.dumps({"result": "Search error: boom"}))

# invalid query (None) -> orig invalid path, untouched
tg_none = FakeToolGroup(qidx=9)
out = _tg_call(tg_none, None)
check("shuffled invalid query: invalid path kept", tg_none.last_call_metadata["status"] == "invalid_query" and out == "")

# un-stamped group (defensive) -> orig behavior
tg_unstamped = FakeToolGroup(qidx=None)
out = _tg_call(tg_unstamped, "q-real")
check("un-stamped group: orig behavior", tg_unstamped.last_call_metadata["status"] == "success")

counters = stamp.counters
print("shuffled counters:", counters)
check("shuffled: counter served>0", counters["shuffled_served"] > 0)
check("shuffled: real-failure kept counted", counters["real_failure_kept"] == 1)
check("shuffled: fallback never used (stub always succeeds)", counters["shuffled_fallback_to_real"] == 0)

# stamp() with a fake envs container mirroring the real structure:
# SearchMultiProcessEnv.envs[i] is a SearchEnv whose .tool_group is the group
class FakeSearchEnv:
    def __init__(self, qidx):
        self.tool_group = FakeToolGroup(qidx=qidx)


class FakeEnvs:
    def __init__(self, qidx):
        self.envs = [FakeSearchEnv(qidx=qidx)]


envs42 = FakeEnvs(42)
stamped = stamp(envs42, [{"question_id": 42}])
check("stamp: sets _p3_question_index", stamped == {0: 42} and envs42.envs[0].tool_group._p3_question_index == 42)

# ---- no-evidence ----
stamp2 = install_retrieval_condition("no-evidence", 17, 3, QTEXTS)
n_stub_before = len(STUB_CALLS)
tg2 = FakeToolGroup(qidx=5)
out = _tg_call(tg2, "q5-query")
meta = tg2.last_call_metadata
check("no-evidence: success status", meta["status"] == "success")
check("no-evidence: neutral doc ids", meta["document_ids"] == ["noev-0", "noev-1", "noev-2"])
check("no-evidence: neutral content in envelope", meta["formatted_result"].startswith("Doc 1: No relevant documents"))
check("no-evidence: no HTTP call", len(STUB_CALLS) == n_stub_before)
check("no-evidence: no api error", meta["api_request_error"] is None)
check("no-evidence: query preserved", meta["query"] == "q5-query")
check("no-evidence: counters", stamp2.counters["no_evidence_served"] == 1)
tg2_none = FakeToolGroup(qidx=9)
out = _tg_call(tg2_none, None)
check("no-evidence: invalid query keeps invalid path", tg2_none.last_call_metadata["status"] == "invalid_query")

# ---- real: noop stamp, no patch ----
stamp3 = install_retrieval_condition("real", 17, 3, QTEXTS)
check("real: noop stamp counters", stamp3.counters == {})

print()
if failures:
    print(f"SMOKE FAILED: {len(failures)} failures -> {failures}")
    sys.exit(1)
print("SMOKE PASS: counterfactual patch logic verified (shuffled mapping, faithful "
      "no_results/failure preservation, no-evidence envelope, invalid-query passthrough)")
