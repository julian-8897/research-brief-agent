import logging
import secrets
import time
import uuid
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
from src.llm import build_llm_provider
from src.models import BriefRequest, IngestRequest, IngestResponse, SearchResponse
from src.observability import RunRecordStore, Tracer, configure_logging, log_event
from src.retrieval import InMemoryVectorStore, PaperVectorStore, build_vector_store
from src.settings import Settings, get_settings

logger = logging.getLogger("research_brief.api")


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
    run_records: RunRecordStore | None = None
    vector_store_status: VectorStoreStatus | None = None

    def __post_init__(self) -> None:
        if self.run_records is None:
            self.run_records = RunRecordStore(
                self.settings.run_records_dir,
                required=self.settings.run_records_required,
            )
        if self.vector_store_status is None:
            self.vector_store_status = VectorStoreStatus(
                backend=self.vector_store.backend_name
            )
        # Built lazily and cached so search-time query expansion reuses one
        # provider client instead of reconstructing it per request.
        self._llm_provider = None
        self._llm_provider_built = False

    @property
    def llm_provider(self):
        if not self._llm_provider_built:
            self._llm_provider = build_llm_provider(self.settings)
            self._llm_provider_built = True
        return self._llm_provider

    @property
    def tools(self) -> ResearchTools:
        return ResearchTools(
            settings=self.settings,
            arxiv_client=self.arxiv_client,
            embedder=self.embedder,
            vector_store=self.vector_store,
            llm=self.llm_provider,
        )

    @property
    def agent(self) -> ResearchBriefAgent:
        return ResearchBriefAgent(
            settings=self.settings,
            tools=self.tools,
            tracer=self.tracer,
            llm=self.llm_provider,
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
        run_records=RunRecordStore(
            settings.run_records_dir,
            required=settings.run_records_required,
        ),
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


def _final_summary(event: dict) -> dict[str, object]:
    data = event.get("data")
    if not isinstance(data, dict):
        return {}
    usage = data.get("token_cost_estimate") or {}
    retrieval = data.get("retrieval_diagnostics") or {}
    full_text = data.get("full_text_diagnostics") or {}
    return {
        "brief_latency_ms": data.get("latency_ms"),
        "warnings": data.get("warnings", []),
        "cited_papers": len(data.get("cited_papers", [])),
        "retrieved": retrieval.get("returned"),
        "full_text_attempted": full_text.get("attempted"),
        "full_text_succeeded": full_text.get("succeeded"),
        "llm_call_count": usage.get("llm_call_count"),
        "tool_call_count": usage.get("tool_call_count"),
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "estimated_cost_usd": usage.get("estimated_cost_usd"),
        "langfuse_trace_url": data.get("langfuse_trace_url"),
    }


def _log_brief_event(
    event: dict,
    *,
    request_id: str,
    run_id: str,
    provider: str,
    model: str | None,
) -> None:
    event_type = str(event.get("event"))
    payload = {
        "request_id": request_id,
        "run_id": run_id,
        "provider": provider,
        "model": model,
        "agent_event": event_type,
    }
    for key in (
        "turn",
        "name",
        "code",
        "reason",
        "stage",
        "returned",
        "latency_ms",
        "llm_calls",
        "full_text_fetched",
    ):
        if key in event:
            payload[key] = event[key]
    if event_type == "final":
        payload.update(_final_summary(event))
    log_event(logger, "brief_event", **payload)


def create_app(initial_services: Services | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(service_app: FastAPI):
        services = initial_services or create_services(get_settings())
        configure_logging(services.settings)
        service_app.state.services = services
        service_app.state.rate_limiter = InProcessRateLimiter(
            services.settings.rate_limit_requests,
            services.settings.rate_limit_window_seconds,
        )
        try:
            yield
        finally:
            services.tracer.flush()

    service_app = FastAPI(
        title="Research Brief Agent",
        version="0.1.0",
        description=(
            "LLM-powered cited research briefs for AI/ML and scientific-ML "
            "engineering decisions, backed by persistent arXiv retrieval."
        ),
        lifespan=lifespan,
    )

    static_dir = Path(__file__).resolve().parents[2] / "static"
    if static_dir.exists():
        service_app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @service_app.middleware("http")
    async def protect_public_endpoints(request: Request, call_next):
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        request.state.request_id = request_id
        started = time.perf_counter()
        if request.url.path == "/health":
            response = await call_next(request)
            response.headers["X-Request-ID"] = request_id
            log_event(
                logger,
                "http_request",
                request_id=request_id,
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                latency_ms=(time.perf_counter() - started) * 1000,
            )
            return response

        services = get_services(request)
        settings = services.settings
        if not _authorized(request, settings):
            log_event(
                logger,
                "http_request_unauthorized",
                request_id=request_id,
                method=request.method,
                path=request.url.path,
                status_code=401,
            )
            return JSONResponse(
                status_code=401,
                content={
                    "detail": (
                        f"Missing or invalid {settings.api_key_header_name} header."
                    )
                },
                headers={"X-Request-ID": request_id},
            )

        limiter: InProcessRateLimiter = request.app.state.rate_limiter
        allowed, _remaining = limiter.allow(_request_ip(request))
        if not allowed:
            log_event(
                logger,
                "http_request_rate_limited",
                request_id=request_id,
                method=request.method,
                path=request.url.path,
                status_code=429,
            )
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded. Try again later."},
                headers={
                    "Retry-After": str(settings.rate_limit_window_seconds),
                    "X-Request-ID": request_id,
                },
            )

        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        log_event(
            logger,
            "http_request",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            latency_ms=(time.perf_counter() - started) * 1000,
        )
        return response

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
                sort=services.settings.arxiv_sort,
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
        expand: bool | None = Query(
            default=None,
            description="Override LLM query expansion (default from settings).",
        ),
        backfill: bool | None = Query(
            default=None,
            description="Override arXiv coverage backfill (default from settings).",
        ),
    ) -> SearchResponse:
        return services.tools.semantic_search(
            query, k, expand=expand, backfill=backfill
        )

    @service_app.post("/briefs/stream")
    async def stream_brief(
        request: BriefRequest,
        raw_request: Request,
        services: Annotated[Services, Depends(get_services)],
    ) -> StreamingResponse:
        run_id = uuid.uuid4().hex
        request_id = getattr(raw_request.state, "request_id", run_id)
        provider = services.llm_provider
        provider_name = (
            provider.name if provider is not None else services.settings.llm_provider
        )
        provider_model = provider.model if provider is not None else None
        started = time.perf_counter()
        services.run_records.start(
            run_id,
            {
                "request_id": request_id,
                "request": request.model_dump(mode="json"),
                "provider": provider_name,
                "model": provider_model,
                "vector_store": services.vector_store.backend_name,
                "environment": services.settings.environment,
            },
        )
        log_event(
            logger,
            "brief_run_started",
            request_id=request_id,
            run_id=run_id,
            provider=provider_name,
            model=provider_model,
            vector_store=services.vector_store.backend_name,
        )

        async def _events():
            status = "ended_without_final"
            final_summary: dict[str, object] = {}
            try:
                async for event in services.agent.stream(request):
                    enriched = {
                        **event,
                        "run_id": run_id,
                        "request_id": request_id,
                    }
                    services.run_records.event(
                        run_id,
                        {
                            "request_id": request_id,
                            "event": enriched,
                        },
                    )
                    _log_brief_event(
                        enriched,
                        request_id=request_id,
                        run_id=run_id,
                        provider=provider_name,
                        model=provider_model,
                    )
                    if enriched.get("event") == "final":
                        status = "completed"
                        final_summary = _final_summary(enriched)
                    yield format_sse_event(enriched)
            except Exception as exc:
                status = "error"
                services.run_records.event(
                    run_id,
                    {
                        "request_id": request_id,
                        "event": {
                            "event": "stream_exception",
                            "message": str(exc),
                            "type": type(exc).__name__,
                        },
                    },
                )
                log_event(
                    logger,
                    "brief_run_error",
                    request_id=request_id,
                    run_id=run_id,
                    error=str(exc),
                    error_type=type(exc).__name__,
                )
                raise
            finally:
                latency_ms = (time.perf_counter() - started) * 1000
                services.run_records.finish(
                    run_id,
                    {
                        "request_id": request_id,
                        "status": status,
                        "latency_ms": latency_ms,
                        **final_summary,
                    },
                )
                log_event(
                    logger,
                    "brief_run_finished",
                    request_id=request_id,
                    run_id=run_id,
                    status=status,
                    latency_ms=latency_ms,
                    **final_summary,
                )

        return StreamingResponse(
            _events(),
            media_type="text/event-stream",
            headers={"X-Run-ID": run_id, "X-Request-ID": request_id},
        )

    return service_app


app = create_app()
