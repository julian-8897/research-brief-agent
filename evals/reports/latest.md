# Research Brief Evaluation

| Case | Latency ms | Retrieval ms | LLM calls | Est. cost | Citations | Trace |
|---|---:|---:|---:|---:|---:|---|
| rag_science | 37968 | 1 | 7 | $0.08487 | 3 | no |
| astro_instrumentation | 25258 | 0 | 4 | $0.05415 | 2 | no |
| ml_uncertainty | 18752 | 1 | 7 | $0.06900 | 3 | no |

Judge checks to run on reviewed outputs:
- Answer relevance to the research question.
- Citation grounding against supplied titles and abstracts.
- Unsupported-claim risk.
- Useful uncertainty or refusal behavior when evidence is weak.
