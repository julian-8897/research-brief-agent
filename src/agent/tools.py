import time
from dataclasses import dataclass

from src.agent.query_expansion import expand_query
from src.arxiv_client import ArxivClient
from src.embeddings import TextEmbedder, build_paper_embedding_text
from src.ingestion import fetch_arxiv_papers
from src.llm import LLMProvider
from src.models import (
    BriefRequest,
    RetrievalDiagnostics,
    SearchResponse,
    SearchResponseItem,
)
from src.retrieval import PaperVectorStore
from src.settings import Settings


@dataclass
class RetrievalResult:
    items: list[SearchResponseItem]
    diagnostics: RetrievalDiagnostics


class ResearchTools:
    def __init__(
        self,
        settings: Settings,
        arxiv_client: ArxivClient,
        embedder: TextEmbedder,
        vector_store: PaperVectorStore,
        llm: LLMProvider | None = None,
    ):
        self.settings = settings
        self.arxiv_client = arxiv_client
        self.embedder = embedder
        self.vector_store = vector_store
        self.llm = llm

    def arxiv_metadata_search(self, request: BriefRequest):
        domain_part = (
            f" AND cat:{request.domain}"
            if request.domain and "." in request.domain
            else ""
        )
        query = f"all:{request.research_question}{domain_part}"
        return fetch_arxiv_papers(
            self.arxiv_client,
            query=query,
            max_papers=min(request.max_papers, self.settings.max_ingest_results),
            date_range=request.date_range,
            sort=self.settings.arxiv_sort,
        )

    def vector_retrieve(
        self, query: str, k: int, *, embed_text: str | None = None
    ) -> RetrievalResult:
        started = time.perf_counter()
        query_embedding = self.embedder.encode_queries(
            [embed_text or query], batch_size=self.settings.embedding_batch_size
        )[0]
        items = self.vector_store.search(
            query_embedding, k=min(k, self.settings.max_retrieval_results)
        )
        if self.settings.retrieval_min_score > 0.0:
            items = [
                item
                for item in items
                if item.score >= self.settings.retrieval_min_score
            ]
        latency_ms = (time.perf_counter() - started) * 1000
        scores = [item.score for item in items]
        return RetrievalResult(
            items=items,
            diagnostics=RetrievalDiagnostics(
                query=query,
                requested_k=k,
                returned=len(items),
                retrieval_latency_ms=latency_ms,
                min_score=min(scores) if scores else None,
                max_score=max(scores) if scores else None,
            ),
        )

    def fetch_and_ingest(self, query: str, max_papers: int, date_range=None):
        """Fetch fresh arXiv metadata for a model-supplied query and index it.

        Returns ``(ingested_count, papers)`` so the agent loop can report what
        the corpus gained before re-running semantic search.
        """
        papers = fetch_arxiv_papers(
            self.arxiv_client,
            query=query,
            max_papers=min(max_papers, self.settings.max_ingest_results),
            date_range=date_range,
            sort=self.settings.arxiv_sort,
        )
        return self.ingest_papers(papers), papers

    def ingest_papers(self, papers) -> int:
        if not papers:
            return 0
        texts = [build_paper_embedding_text(paper.model_dump()) for paper in papers]
        embeddings = self.embedder.encode_documents(
            texts, batch_size=self.settings.embedding_batch_size
        )
        return self.vector_store.upsert(papers, embeddings)

    def extract_evidence(self, items: list[SearchResponseItem]) -> list[dict]:
        evidence = []
        for item in items:
            paper = item.paper
            evidence.append(
                {
                    "id": paper.id,
                    "title": paper.title,
                    "authors": paper.authors[:5],
                    "published": (
                        paper.published.date().isoformat() if paper.published else None
                    ),
                    "category": paper.primary_category,
                    "score": item.score,
                    "abstract": paper.summary,
                    "arxiv_url": paper.arxiv_url,
                }
            )
        return evidence

    def semantic_search(
        self,
        query: str,
        k: int,
        *,
        expand: bool | None = None,
        backfill: bool | None = None,
    ) -> SearchResponse:
        """Search the indexed corpus for a raw user query.

        Two coverage-oriented behaviors wrap the bare vector search:

        * expansion — a short keyword query is rewritten (LLM, HyDE-style) into
          a descriptive sentence before embedding, which SPECTER2 ranks better.
        * backfill — fresh arXiv papers for the query are fetched and indexed so
          the endpoint returns relevant literature even when the corpus did not
          already contain it, instead of ranking only whatever was indexed.

        ``expand``/``backfill`` override the configured defaults when set.
        """
        do_expand = (
            self.settings.query_expansion_enabled if expand is None else expand
        )
        do_backfill = (
            self.settings.search_auto_backfill if backfill is None else backfill
        )

        embed_text, expanded = expand_query(
            query,
            self.llm if do_expand else None,
            enabled=do_expand,
            max_words=self.settings.query_expansion_max_words,
        )

        backfilled = 0
        if do_backfill:
            backfilled, _papers = self.fetch_and_ingest(
                query, self.settings.search_backfill_max_papers
            )

        result = self.vector_retrieve(query, k, embed_text=embed_text)
        return SearchResponse(
            query=query,
            results=result.items,
            retrieval_latency_ms=result.diagnostics.retrieval_latency_ms,
            expanded_query=expanded,
            backfilled=backfilled,
        )
