# Research Brief Evaluation

| Case | Status | Latency ms | Retrieval ms | LLM calls | Tool calls | Full text | Est. cost | Citations | Trace |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| rag_briefing | warnings:1 | 0 | 0 | 0 | 0 | 0/0 | $0.00775 | 3 | no |
| instrumentation_ops | warnings:1 | 0 | 0 | 0 | 0 | 0/0 | $0.00781 | 3 | no |
| ml_uncertainty | warnings:1 | 0 | 0 | 0 | 0 | 0/0 | $0.00760 | 3 | no |

## Quality metrics (automated)

| Case | Cited | Valid ids | Halluc. | Read-in-full | Uncertainty | Cite support (LLM) | Faithfulness | Answer rel. |
|---|---:|---:|---:|---:|---|---:|---:|---:|
| rag_briefing | 3 | 100% | 0 | 0% | ok | - | - | - |
| instrumentation_ops | 3 | 100% | 0 | 0% | ok | - | - | - |
| ml_uncertainty | 3 | 100% | 0 | 0% | ok | - | - | - |

## Retrieval relevance

| Case | k | Hits | Recall@k | nDCG@k |
|---|---:|---:|---:|---:|
| rag_briefing | 3 | 1 | 100% | 63% |
| instrumentation_ops | 3 | 1 | 100% | 50% |
| ml_uncertainty | 3 | 1 | 100% | 100% |

## Aggregate

- Cases: 3
- Mean citation grounding: 100%
- Mean hallucination rate: 0% (0 case(s) with fabricated ids)
- Mean cited-papers-read-in-full: 0%
- Mean full-text fetch success: 0%
- Uncertainty signaled appropriately: 100% of cases
