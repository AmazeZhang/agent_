# Aware-v2 vs StepSearch smoke-16 retrieval screen

Dataset SHA256: `c36c2ea42d4d377970914f2042d132977981d6785477f8f51ca349c35de0a495`

| Metric | Aware-v2 | StepSearch |
|---|---:|---:|
| Evidence-hit questions | 8/16 | 5/16 |
| Evidence-hit calls | 8/18 | 8/21 |
| Multi-hop episodes | 2/16 | 5/16 |
| True redundant searches | 1 | 0 |
| Answer compliance | 16/16 | 9/16 |
| EM | 4/16 | 2/16 |

## Paired evidence-hit sets

- `both_hit` (5): [4, 5, 11, 14, 15]
- `stepsearch_only_hit` (0): []
- `aware_only_hit` (3): [0, 3, 10]
- `neither_hit` (8): [1, 2, 6, 7, 8, 9, 12, 13]

## Aware-only evidence-hit cases

- Q0: when did how you remind me come out?
  - Aware query: `when did how you remind me come out`
  - StepSearch queries: ["how you remind me release date"]
- Q3: Where did Henri Christophe and other slaves hold an uprising from 1791 to 1804 that led to the founding of a state which was both free from slavery and ruled by non-whites and former captives?
  - Aware query: `Where did Henri Christophe and other slaves hold an uprising from 1791 to 1804 that led to the founding of a state which was both free from slavery and ruled by non-whites and former captives?`
  - StepSearch queries: ["Henri Christophe uprising 1791-1804"]
- Q10: "Whose memoirs, published in 2010, were called ""A Journey""?"?
  - Aware query: `Whose memoirs published in 2010 were called "A Journey"?`
  - StepSearch queries: ["Memoirs published in 2010"]

## Decision

Eligible for a bounded Aware mechanism experiment: **no**.
The models and prompt protocols differ, so this is a descriptive mechanism screen, not a causal algorithm comparison.
