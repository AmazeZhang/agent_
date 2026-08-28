from searchr1_repro.stepsearch_protocol import (
    STEPSEARCH_SOURCE_COMMIT,
    build_stepsearch_prompt,
    stepsearch_prompt_contains_query,
    truncate_stepsearch_response,
)


def test_initial_prompt_has_official_control_tags_and_question():
    prompt = build_stepsearch_prompt("Who wrote Imagine?")
    assert STEPSEARCH_SOURCE_COMMIT == "43215bab9118a4c8e01b15082f74b2aea30c1fc8"
    assert "<plan>" in prompt
    assert "<search>" in prompt
    assert "<information>" in prompt
    assert "<observation>" in prompt
    assert "<answer>" in prompt
    assert prompt.endswith(" Question: Who wrote Imagine?\n")


def test_followup_prompt_appends_raw_plan_search_and_information_trace():
    trace = (
        "Step 1:<plan>Find the songwriter.</plan>"
        "<search>Imagine songwriter</search> "
        "<information>John Lennon wrote Imagine.</information>\n\n"
    )
    prompt = build_stepsearch_prompt("Who wrote Imagine?", trace)
    assert prompt.endswith(trace)
    assert prompt.count("## Background") == 1
    assert "<plan>Find the songwriter.</plan>" in prompt
    assert "<information>John Lennon wrote Imagine.</information>" in prompt


def test_response_is_clipped_at_first_search_before_projection_and_memory():
    raw = (
        "<plan>Find the songwriter.</plan><search>Imagine songwriter </search>"
        "<information>fabricated</information><answer>fabricated</answer>"
    )
    clipped = truncate_stepsearch_response(raw)
    assert clipped == "<plan>Find the songwriter.</plan><search>Imagine songwriter </search>"
    assert "fabricated" not in clipped


def test_response_without_search_is_clipped_at_first_answer():
    raw = "<plan>I know it.</plan><answer>John Lennon</answer>trailing junk"
    assert truncate_stepsearch_response(raw) == "<plan>I know it.</plan><answer>John Lennon</answer>"


def test_prompt_query_check_tolerates_tag_whitespace_but_requires_exact_query():
    prompt = "trace <search>  Imagine songwriter \n</search> <information>real</information>"
    assert stepsearch_prompt_contains_query(prompt, "Imagine songwriter")
    assert not stepsearch_prompt_contains_query(prompt, "Imagine release date")
