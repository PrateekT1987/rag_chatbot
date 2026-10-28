"""End-to-end API tests (no real LLM, no network)."""
from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from backend.config import Settings
from backend.main import create_app

from conftest import CATALOG_CSV, FakeLLM, NoLLM


def build_client(**overrides) -> TestClient:
    settings = Settings()
    settings.catalog_path = Path(CATALOG_CSV)
    for key, value in overrides.items():
        setattr(settings, key, value)
    llm = overrides.pop("llm", None)
    app = create_app(settings=settings, llm=llm or FakeLLM())
    return TestClient(app)


def test_health_reports_catalog_size():
    with build_client() as client:
        data = client.get("/health").json()
    assert data["status"] == "ok"
    assert data["products"] >= 50
    assert data["llm_configured"] is True


def test_health_without_llm_key():
    settings = Settings()
    settings.llm_api_key = ""
    app = create_app(settings=settings, llm=NoLLM())
    with TestClient(app) as client:
        data = client.get("/health").json()
    assert data["llm_configured"] is False


def test_products_endpoint():
    with build_client() as client:
        products = client.get("/api/products").json()
    assert len(products) >= 50
    skus = {p["sku"] for p in products}
    assert "AUR-H700" in skus
    assert all("name" in p and "price" in p for p in products)


def test_chat_roundtrip_and_product_cards():
    llm = FakeLLM(reply="The AUR-H700 costs 129.99 USD.")
    with build_client(llm=llm) as client:
        res = client.post(
            "/api/chat",
            json={"message": "how much are the Aurora Wireless Headphones?"},
        )
    assert res.status_code == 200
    body = res.json()
    assert body["fallback"] is False
    assert body["answer"] == "The AUR-H700 costs 129.99 USD."
    # Both the base model and its Lite variant are legitimately retrieved;
    # the LLM disambiguates. The named product must be among the cards.
    assert any(p["sku"] == "AUR-H700" for p in body["products"])
    assert all("name" in p and "price" in p for p in body["products"])
    h700 = next(p for p in body["products"] if p["sku"] == "AUR-H700")
    assert h700["price"] == 129.99


def test_chat_fallback_for_off_topic():
    llm = FakeLLM()
    with build_client(llm=llm) as client:
        body = client.post("/api/chat", json={"message": "capital of france"}).json()
    assert body["fallback"] is True
    assert body["products"] == []
    assert llm.calls == []


def test_chat_validates_message_length():
    with build_client() as client:
        res = client.post("/api/chat", json={"message": ""})
    assert res.status_code == 422


def test_rate_limit_returns_429():
    with build_client(rate_limit_per_minute=3) as client:
        for _ in range(3):
            assert client.post("/api/chat", json={"message": "smart plug"}).status_code == 200
        res = client.post("/api/chat", json={"message": "smart plug"})
    assert res.status_code == 429
    assert "Too many requests" in res.json()["detail"]


def test_reindex_endpoint_requires_token():
    with build_client(admin_token="secret123") as client:
        assert client.post("/api/reindex").status_code == 401
        assert client.post(
            "/api/reindex", headers={"X-Admin-Token": "wrong"}
        ).status_code == 401
        ok = client.post("/api/reindex", headers={"X-Admin-Token": "secret123"})
    assert ok.status_code == 200
    assert ok.json()["products"] >= 50


def test_reindex_rejects_broken_csv_and_keeps_old_catalog(tmp_path):
    import shutil

    csv = tmp_path / "products.csv"
    shutil.copy(CATALOG_CSV, csv)
    settings = Settings()
    settings.catalog_path = csv
    app = create_app(settings=settings, llm=FakeLLM())

    with TestClient(app) as client:
        csv.write_text("name,price\nBroken,1.00\n")  # missing required columns
        res = client.post("/api/reindex")
        assert res.status_code == 400
        assert "missing required column" in res.json()["detail"]
        # Old catalog still served.
        assert client.get("/health").json()["products"] >= 50


def test_widget_and_demo_are_served():
    with build_client() as client:
        js = client.get("/widget.js")
        page = client.get("/")
    assert js.status_code == 200
    assert "javascript" in js.headers["content-type"]
    assert "pc-launcher" in js.text
    assert page.status_code == 200
    assert "widget.js" in page.text


def test_cors_allows_configured_origin():
    with build_client(cors_origins=["https://shop.example.com"]) as client:
        res = client.get(
            "/health", headers={"Origin": "https://shop.example.com"}
        )
    assert res.headers.get("access-control-allow-origin") == "https://shop.example.com"


def test_no_cors_headers_when_origins_empty():
    """Default is restrictive: no cross-origin access unless configured."""
    with build_client(cors_origins=[]) as client:
        res = client.get("/health", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in res.headers


def test_reindex_restricted_to_localhost_without_token():
    """Without ADMIN_TOKEN, reindex must not be reachable remotely."""
    settings = Settings()
    settings.catalog_path = Path(CATALOG_CSV)
    settings.admin_token = ""
    app = create_app(settings=settings, llm=FakeLLM())
    with TestClient(app, client=("203.0.113.9", 40000)) as client:
        res = client.post("/api/reindex")
    assert res.status_code == 403
    assert "localhost" in res.json()["detail"]

