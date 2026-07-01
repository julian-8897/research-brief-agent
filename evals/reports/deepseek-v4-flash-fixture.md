# Research Brief Evaluation

| Case | Status | Latency ms | Retrieval ms | LLM calls | Tool calls | Full text | Est. cost | Citations | Trace |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| rag_science | ok | 25902 | 1 | 6 | 6 | 2/2 | $0.07216 | 2 | no |
| astro_instrumentation | ok | 22773 | 1 | 6 | 5 | 2/2 | $0.06798 | 2 | no |
| ml_uncertainty | ok | 25734 | 1 | 5 | 6 | 3/3 | $0.07687 | 3 | no |

Judge checks to run on reviewed outputs:
- Answer relevance to the research question.
- Citation grounding against supplied titles and abstracts.
- Unsupported-claim risk.
- Useful uncertainty or refusal behavior when evidence is weak.
