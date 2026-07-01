import secrets
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Annotated

import arxiv
import requests
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from src.agent import ResearchBriefAgent, ResearchTools
from src.api.sse import format_sse_event
from src.arxiv_client import ArxivClient
from src.embeddings import TextEmbedder, build_paper_embedding_text
from src.ingestion import fetch_arxiv_papers
from src.models import BriefRequest, IngestRequest, IngestResponse, SearchResponse
from src.observability import Tracer
from src.retrieval import InMemoryVectorStore, PaperVectorStore, build_vector_store
from src.settings import Settings, get_settings


@dataclass(frozen=True)
class VectorStoreStatus:
    backend: str
    fallback: bool = False
    error: str | None = None


@dataclass
class Services:
    settings: Settings
    arxiv_client: ArxivClient
    embedder: TextEmbedder
    vector_store: PaperVectorStore
    tracer: Tracer
    vector_store_status: VectorStoreStatus | None = None

    def __post_init__(self) -> None:
        if self.vector_store_status is None:
            self.vector_store_status = VectorStoreStatus(
                backend=self.vector_store.backend_name
            )

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


class InProcessRateLimiter:
    def __init__(self, requests: int, window_seconds: int):
        self.requests = requests
        self.window_seconds = window_seconds
        self._hits: dict[str, list[float]] = {}
        self._lock = Lock()

    @property
    def enabled(self) -> bool:
        return self.requests > 0 and self.window_seconds > 0

    def allow(self, key: str, now: float | None = None) -> tuple[bool, int]:
        if not self.enabled:
            return True, self.requests
        now = now or time.monotonic()
        cutoff = now - self.window_seconds
        with self._lock:
            hits = [hit for hit in self._hits.get(key, []) if hit > cutoff]
            if len(hits) >= self.requests:
                self._hits[key] = hits
                return False, 0
            hits.append(now)
            self._hits[key] = hits
            return True, max(self.requests - len(hits), 0)


def _require_production_security(settings: Settings) -> None:
    if settings.environment.lower() == "local":
        return
    if not settings.api_keys:
        raise RuntimeError(
            "API_KEYS must be configured when ENVIRONMENT is not 'local'."
        )


def create_services(settings: Settings) -> Services:
    _require_production_security(settings)
    try:
        vector_store = build_vector_store(settings)
        vector_store.ensure_collection()
        vector_store_status = VectorStoreStatus(backend=vector_store.backend_name)
    except Exception as exc:
        if settings.environment.lower() != "local":
            raise
        vector_store = InMemoryVectorStore(settings.embedding_dimension)
        vector_store_status = VectorStoreStatus(
            backend=vector_store.backend_name,
            fallback=True,
            error=str(exc),
        )
    return Services(
        settings=settings,
        arxiv_client=ArxivClient(),
        embedder=TextEmbedder(
            settings.embedding_model,
            document_adapter=settings.embedding_document_adapter,
            query_adapter=settings.embedding_query_adapter,
        ),
        vector_store=vector_store,
        tracer=Tracer(settings),
        vector_store_status=vector_store_status,
    )


def get_services(request: Request) -> Services:
    return request.app.state.services


def _request_ip(request: Request) -> str:
    fly_client_ip = request.headers.get("fly-client-ip")
    if fly_client_ip:
        return fly_client_ip
    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        return forwarded_for.split(",", maxsplit=1)[0].strip()
    if request.client:
        return request.client.host
    return "unknown"


def _auth_enabled(settings: Settings) -> bool:
    return bool(settings.api_keys)


def _authorized(request: Request, settings: Settings) -> bool:
    if not _auth_enabled(settings):
        return True
    supplied = request.headers.get(settings.api_key_header_name)
    if not supplied:
        return False
    return any(secrets.compare_digest(supplied, key) for key in settings.api_keys)


def _llm_key_present(settings: Settings) -> bool:
    provider = settings.llm_provider.lower()
    if provider == "anthropic":
        return bool(settings.anthropic_api_key)
    if provider == "openai":
        return bool(settings.openai_api_key)
    return False


def create_app(initial_services: Services | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(service_app: FastAPI):
        services = initial_services or create_services(get_settings())
        service_app.state.services = services
        service_app.state.rate_limiter = InProcessRateLimiter(
            services.settings.rate_limit_requests,
            services.settings.rate_limit_window_seconds,
        )
        yield

    service_app = FastAPI(
        title="Research Brief Agent",
        version="0.1.0",
        description="LLM-powered cited research briefs over persistent arXiv retrieval.",
        lifespan=lifespan,
    )

    static_dir = Path(__file__).resolve().parents[2] / "static"
    if static_dir.exists():
        service_app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @service_app.middleware("http")
    async def protect_public_endpoints(request: Request, call_next):
        if request.url.path == "/health":
            return await call_next(request)

        services = get_services(request)
        settings = services.settings
        if not _authorized(request, settings):
            return JSONResponse(
                status_code=401,
                content={
                    "detail": (
                        f"Missing or invalid {settings.api_key_header_name} header."
                    )
                },
            )

        limiter: InProcessRateLimiter = request.app.state.rate_limiter
        allowed, _remaining = limiter.allow(_request_ip(request))
        if not allowed:
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded. Try again later."},
                headers={"Retry-After": str(settings.rate_limit_window_seconds)},
            )

        return await call_next(request)

    @service_app.get("/")
    def index() -> FileResponse:
        index_path = static_dir / "index.html"
        if not index_path.exists():
            raise HTTPException(status_code=404, detail="static UI not found")
        return FileResponse(index_path)

    @service_app.get("/health")
    def health(services: Annotated[Services, Depends(get_services)]):
        vector_store = {
            "status": "ok",
            "backend": services.vector_store.backend_name,
            "configured_backend": services.settings.vector_store_backend,
            "collection": services.settings.qdrant_collection,
            "fallback": services.vector_store_status.fallback,
            "error": services.vector_store_status.error,
            "papers_indexed": None,
        }
        try:
            vector_store["papers_indexed"] = services.vector_store.count()
        except Exception as exc:
            vector_store["status"] = "error"
            vector_store["error"] = str(exc)

        llm_key_present = _llm_key_present(services.settings)
        llm_provider = {
            "status": "configured" if llm_key_present else "missing_key",
            "provider": services.settings.llm_provider,
            "key_present": llm_key_present,
            "reachable": "not_probed",
        }

        ready = (
            vector_store["status"] == "ok"
            and not services.vector_store_status.fallback
            and llm_key_present
        )
        payload = {
            "status": "ok" if ready else "degraded",
            "ready": ready,
            "app": services.settings.app_name,
            "environment": services.settings.environment,
            "retrieval_backend": services.vector_store.backend_name,
            "collection": services.settings.qdrant_collection,
            "papers_indexed": vector_store["papers_indexed"],
            "vector_store": vector_store,
            "llm_provider": llm_provider,
            "langfuse_enabled": bool(
                services.settings.langfuse_public_key
                and services.settings.langfuse_secret_key
            ),
        }
        return JSONResponse(status_code=200 if ready else 503, content=payload)

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
            embeddings = services.embedder.encode_documents(
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
                yield format_sse_event(event)

        return StreamingResponse(_events(), media_type="text/event-stream")

    return service_app


app = create_app()
