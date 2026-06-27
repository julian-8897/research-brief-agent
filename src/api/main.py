import json
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import arxiv
import requests
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from src.agent import ResearchBriefAgent, ResearchTools
from src.arxiv_client import ArxivClient
from src.embeddings import TextEmbedder, build_paper_embedding_text
from src.ingestion import fetch_arxiv_papers
from src.models import BriefRequest, IngestRequest, IngestResponse, SearchResponse
from src.observability import Tracer
from src.retrieval import InMemoryVectorStore, PaperVectorStore, build_vector_store
from src.settings import Settings, get_settings


@dataclass
class Services:
    settings: Settings
    arxiv_client: ArxivClient
    embedder: TextEmbedder
    vector_store: PaperVectorStore
    tracer: Tracer

    @property
    def tools(self) -> ResearchTools:
        return ResearchTools(
            settings=self.settings,
            arxiv_client=self.arxiv_client,
            embedder=self.embedder,
            vector_store=self.vector_store,
        )

    @property
    def agent(self) -> ResearchBriefAgent:
        return ResearchBriefAgent(
            settings=self.settings,
            tools=self.tools,
            tracer=self.tracer,
        )


def create_services(settings: Settings) -> Services:
    try:
        vector_store = build_vector_store(settings)
        vector_store.ensure_collection()
    except Exception:
        vector_store = InMemoryVectorStore(settings.embedding_dimension)
    return Services(
        settings=settings,
        arxiv_client=ArxivClient(),
        embedder=TextEmbedder(settings.embedding_model),
        vector_store=vector_store,
        tracer=Tracer(settings),
    )


def get_services(request: Request) -> Services:
    return request.app.state.services


def create_app(initial_services: Services | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(service_app: FastAPI):
        service_app.state.services = initial_services or create_services(get_settings())
        yield

    service_app = FastAPI(
        title="arXiv Research Brief Agent",
        version="0.1.0",
        description="LLM-powered cited research briefs over persistent arXiv retrieval.",
        lifespan=lifespan,
    )

    static_dir = Path(__file__).resolve().parents[2] / "static"
    if static_dir.exists():
        service_app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @service_app.get("/")
    def index() -> FileResponse:
        index_path = static_dir / "index.html"
        if not index_path.exists():
            raise HTTPException(status_code=404, detail="static UI not found")
        return FileResponse(index_path)

    @service_app.get("/health")
    def health(services: Annotated[Services, Depends(get_services)]):
        return {
            "status": "ok",
            "app": services.settings.app_name,
            "environment": services.settings.environment,
            "retrieval_backend": services.vector_store.backend_name,
            "collection": services.settings.qdrant_collection,
            "papers_indexed": services.vector_store.count(),
            "llm_provider": services.settings.llm_provider,
            "langfuse_enabled": bool(
                services.settings.langfuse_public_key
                and services.settings.langfuse_secret_key
            ),
        }

    @service_app.post("/ingest", response_model=IngestResponse)
    def ingest(
        request: IngestRequest, services: Annotated[Services, Depends(get_services)]
    ) -> IngestResponse:
        started = time.perf_counter()
        max_papers = min(request.max_papers, services.settings.max_ingest_results)
        try:
            papers = fetch_arxiv_papers(
                services.arxiv_client,
                query=request.query,
                max_papers=max_papers,
                date_range=request.date_range,
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid arXiv ingest request: {exc}",
            ) from exc
        except arxiv.ArxivError as exc:
            raise HTTPException(
                status_code=502,
                detail=(
                    "arXiv API request failed. Try removing the date range, reducing "
                    f"max_papers, or simplifying the query. Original error: {exc}"
                ),
            ) from exc
        except requests.RequestException as exc:
            raise HTTPException(
                status_code=502,
                detail=(
                    "Network request to arXiv failed. Try again later, remove the "
                    f"date range, or reduce max_papers. Original error: {exc}"
                ),
            ) from exc
        if papers:
            texts = [build_paper_embedding_text(paper.model_dump()) for paper in papers]
            embeddings = services.embedder.encode_texts(
                texts, batch_size=services.settings.embedding_batch_size
            )
            ingested = services.vector_store.upsert(papers, embeddings)
        else:
            ingested = 0
        return IngestResponse(
            ingested=ingested,
            collection=services.settings.qdrant_collection,
            retrieval_backend=services.vector_store.backend_name,
            latency_ms=(time.perf_counter() - started) * 1000,
        )

    @service_app.get("/papers/search", response_model=SearchResponse)
    def search_papers(
        services: Annotated[Services, Depends(get_services)],
        query: str = Query(..., min_length=3),
        k: int = Query(default=10, ge=1, le=50),
    ) -> SearchResponse:
        return services.tools.semantic_search(query, k)

    @service_app.post("/briefs/stream")
    async def stream_brief(
        request: BriefRequest, services: Annotated[Services, Depends(get_services)]
    ) -> StreamingResponse:
        async def _events():
            async for event in services.agent.stream(request):
                yield f"data: {json.dumps(event)}\n\n"

        return StreamingResponse(_events(), media_type="text/event-stream")

    return service_app


app = create_app()
