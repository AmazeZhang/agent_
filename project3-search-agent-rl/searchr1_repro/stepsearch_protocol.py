"""Frozen evaluation-only prompt adapter for the open-source StepSearch model.

The wording follows StepSearch's public ``scripts/data_process/musi_search.py``
prompt at commit 43215bab9118a4c8e01b15082f74b2aea30c1fc8.  This module
contains no reward or training code.
"""

STEPSEARCH_SOURCE_COMMIT = "43215bab9118a4c8e01b15082f74b2aea30c1fc8"

STEPSEARCH_PREFIX = """## Background
You are a deep AI research assistant with search tool
 You should first think about your research plan or what to search for next.

## Response format
1. You must make search plan inside <plan> and </plan> for in the beginning and after observation.
2. After plan, if you find you lack some knowledge, you can call a search engine by <search> search keyword </search> and it will return the search results between <information> and </information>.
3. You must conduct observation inside <observation> and </observation> for EVERY searched document, e.g.<observation>Based on retrieved inforamtion, Doc1 ....; Doc2...; Doc3 ...;</observation>
4. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer> without detailed illustrations. For example, <answer> Beijing </answer>

Please follow the loop of plan, search, information, observation, plan ... until the you can answer original question.
 Question: {task_description}
"""


def build_stepsearch_prompt(task_description: str, memory_context: str = "") -> str:
    """Render the official prefix followed by the accumulated action/tool trace."""
    prefix = STEPSEARCH_PREFIX.format(task_description=task_description)
    if not memory_context:
        return prefix
    return f"{prefix}{memory_context}"

