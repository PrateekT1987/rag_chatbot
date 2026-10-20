"""FastAPI backend for the RAG product catalog chatbot.

Run locally:      .venv/bin/uvicorn backend.main:app --reload
Run in production: uvicorn backend.main:app --host 0.0.0.0 --port 8000 --workers 1
"""
from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from .catalog import Catalog, CatalogError
from .config import PROJECT_ROOT, Settings
from .llm import LLMClient
from .models import ChatRequest, ChatResponse, Product
from .rag import RagPipeline
from .retriever import build_index

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("api")

STATIC_DIR = PROJECT_ROOT / "static"


class RateLimiter:
    """Minimal in-memory sliding-window limiter (per client IP).

    Good enough for a single-process deployment; for multi-worker or
    multi-host setups put a shared limiter (or nginx limit_req) in front.
    """

    def __init__(self, per_minute: int):
        self.per_minute = per_minute
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def check(self, key: str) -> bool:
        if self.per_minute <= 0:
            return True
        now = time.monotonic()
        window = self._hits[key]
        while window and now - window[0] > 60.0:
            window.popleft()
        if len(window) >= self.per_minute:
            return False
        window.append(now)
        return True


def create_app(
    settings: Settings | None = None,
    catalog: Catalog | None = None,
    index=None,
    llm: LLMClient | None = None,
) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        if llm is not None:
            await llm.close()

    app = FastAPI(title="RAG Product Catalog Chatbot", version="1.0.0",
                  lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    # -- wiring (dependencies can be injected for tests) -------------------
    if catalog is None:
        try:
            catalog = Catalog(settings.catalog_path)
        except CatalogError as exc:
            log.error("Catalog failed to load: %s", exc)
            raise
    if index is None:
        index = build_index(settings, catalog)
    if llm is None:
        llm = LLMClient(settings)
    pipeline = RagPipeline(index=index, catalog=catalog, llm=llm, settings=settings)
    limiter = RateLimiter(settings.rate_limit_per_minute)

    app.state.settings = settings
    app.state.catalog = catalog
    app.state.index = index
    app.state.pipeline = pipeline

    async def enforce_rate_limit(request: Request) -> None:
        client_ip = request.client.host if request.client else "unknown"
        if not limiter.check(client_ip):
            raise HTTPException(
                status_code=429,
                detail="Too many requests. Please wait a moment and try again.",
            )

    def require_admin(x_admin_token: str = Header(default="")) -> None:
        if not settings.admin_token:
            return  # admin endpoint disabled when no token is configured
        if x_admin_token != settings.admin_token:
            raise HTTPException(status_code=401, detail="Invalid admin token.")

    # -- routes -------------------------------------------------------------

    @app.get("/health")
    async def health() -> dict:
        return {
            "status": "ok",
            "products": index.size,
            "retriever": settings.retriever,
            "llm_configured": llm.available,
            "catalog": str(settings.catalog_path),
        }

    @app.get("/api/products", response_model=list[Product])
    async def list_products() -> list[Product]:
        catalog.load()  # pick up CSV edits
        return catalog.products

    @app.post("/api/chat", response_model=ChatResponse,
              dependencies=[Depends(enforce_rate_limit)])
    async def chat(request: ChatRequest) -> ChatResponse:
        return await pipeline.answer(request)

    @app.post("/api/reindex", dependencies=[Depends(require_admin)])
    async def reindex() -> JSONResponse:
        try:
            catalog.load(force=True)
            index.rebuild()
        except (CatalogError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return JSONResponse(
            {"status": "ok", "products": index.size}
        )

    # Widget + demo page served from the same origin as the API.
    @app.get("/widget.js")
    async def widget_js() -> FileResponse:
        return FileResponse(STATIC_DIR / "chat-widget.js",
                            media_type="application/javascript")

    @app.get("/", dependencies=[Depends(enforce_rate_limit)])
    async def demo() -> FileResponse:
        return FileResponse(STATIC_DIR / "demo.html")

    return app


app = create_app()
