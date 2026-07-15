# Research Brief Evaluation

| Case | Status | Latency ms | Retrieval ms | LLM calls | Tool calls | Full text | Est. cost | Citations | Trace |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| agentic_rag | ok | 29015 | 2 | 5 | 6 | 3/3 | $0.11381 | 3 | no |
| rag_vs_finetune_vs_longcontext | ok | 30930 | 1 | 6 | 7 | 3/3 | $0.12023 | 4 | no |
| quantization_inference | ok | 25842 | 1 | 6 | 6 | 1/1 | $0.08817 | 1 | no |
| adam_vs_sgd | ok | 26948 | 1 | 7 | 7 | 1/1 | $0.09590 | 5 | no |
| compute_optimal_scaling | ok | 25891 | 1 | 8 | 7 | 1/1 | $0.12009 | 6 | no |
| pinns_vs_solvers | ok | 26745 | 1 | 6 | 5 | 2/2 | $0.09034 | 2 | no |
| uq_ensembles_vs_gp | ok | 29829 | 1 | 6 | 8 | 3/3 | $0.12614 | 3 | no |

## Quality metrics (automated)

| Case | Cited | Valid ids | Halluc. | Read-in-full | Uncertainty | Cite support (LLM) | Faithfulness | Answer rel. |
|---|---:|---:|---:|---:|---|---:|---:|---:|
| agentic_rag | 3 | 100% | 0 | 100% | ok | - | - | - |
| rag_vs_finetune_vs_longcontext | 4 | 100% | 0 | 75% | ok | - | - | - |
| quantization_inference | 1 | 100% | 0 | 100% | ok | - | - | - |
| adam_vs_sgd | 5 | 100% | 0 | 20% | ok | - | - | - |
| compute_optimal_scaling | 6 | 100% | 0 | 17% | ok | - | - | - |
| pinns_vs_solvers | 2 | 100% | 0 | 100% | ok | - | - | - |
| uq_ensembles_vs_gp | 3 | 100% | 0 | 100% | ok | - | - | - |

## Retrieval relevance

| Case | k | Hits | Recall@k | nDCG@k |
|---|---:|---:|---:|---:|
| agentic_rag | 6 | 1 | 33% | 47% |
| rag_vs_finetune_vs_longcontext | 6 | 1 | 33% | 47% |
| quantization_inference | 6 | 1 | 33% | 47% |
| adam_vs_sgd | 6 | 1 | 33% | 47% |
| compute_optimal_scaling | 6 | 1 | 33% | 47% |
| pinns_vs_solvers | 6 | 1 | 33% | 47% |
| uq_ensembles_vs_gp | 6 | 1 | 33% | 47% |

## Aggregate

- Cases: 7
- Mean citation grounding: 100%
- Mean hallucination rate: 0% (0 case(s) with fabricated ids)
- Mean cited-papers-read-in-full: 73%
- Mean full-text fetch success: 100%
- Uncertainty signaled appropriately: 100% of cases
