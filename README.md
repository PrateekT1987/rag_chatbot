# RAG Product Catalog Chatbot

A lightweight Retrieval-Augmented Generation chatbot that answers customer
questions (pricing, specs, availability, SKUs, comparisons) from a product
CSV — with strict guards against hallucinated answers.

**Stack:** Python / FastAPI · TF-IDF vector search (scikit-learn) ·
Any OpenAI-compatible LLM · Dependency-free JS chat widget.

```
┌──────────────┐   POST /api/chat    ┌─────────────────────────────────────┐
│  Chat widget │ ──────────────────► │  FastAPI backend                    │
│  (any HTML   │ ◄────────────────── │  1. vector-search catalog CSV (k=4) │
│   page)      │  answer + products  │  2. threshold check → else fallback │
└──────────────┘                     │  3. LLM answer from CSV data only   │
                                     │  4. SKU hallucination guard         │
                                     └──────────────┬──────────────────────┘
                                                    │
                              data/products.csv ◄───┘   LLM API (optional)
```

Design priorities: **accuracy, simplicity, maintainability**. No vector
database, no message queue, no embedding service required — ~50 products are
indexed in milliseconds in-process.

---

## 1. Quick start (local)

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

cp .env.example .env          # add your LLM_API_KEY (optional but recommended)
.venv/bin/uvicorn backend.main:app --reload
```

Open <http://localhost:8000> — a demo storefront with the widget embedded.

- Without an LLM key the bot still works: it lists matching catalog products
  instead of generating prose ("no-LLM mode").
- The LLM works with any OpenAI-compatible endpoint: OpenAI, Groq, Together,
  OpenRouter, or a local server (Ollama, vLLM, LM Studio) via `LLM_BASE_URL`.

### If you received this project as a delivery

1. Copy `.env.example` to `.env` and put **your own** API key in `LLM_API_KEY`
   — never reuse a key someone else shared with you, and never paste keys
   into HTML/JavaScript (the widget only talks to this backend; keys stay
   server-side).
2. You can demo the bot immediately **without any key** (no-LLM mode).
3. Replace `data/products.csv` with your catalog (same columns).
4. Run the test suite once: `.venv/bin/python -m pytest`.

## 2. Configuration (`.env`)

All configuration is environment variables (see `.env.example` for the full
list with comments). Highlights:

| Variable | Default | Purpose |
|---|---|---|
| `LLM_API_KEY` | — | Enables generated answers. Also used by `RETRIEVER=openai`. |
| `LLM_BASE_URL` | `https://api.openai.com/v1` | Any OpenAI-compatible endpoint. |
| `LLM_MODEL` | `gpt-4o-mini` | Chat model name. |
| `CATALOG_PATH` | `data/products.csv` | Location of the product CSV. |
| `RETRIEVER` | `tfidf` | `tfidf` (no infrastructure) or `openai` (embeddings). |
| `TOP_K` | `4` | How many products the LLM sees per question. |
| `RELEVANCE_THRESHOLD` | `0.10` | Below this similarity the bot refuses (fallback) instead of guessing. |
| `CORS_ORIGINS` | `*` | Comma-separated site origins allowed to call the API. |
| `RATE_LIMIT_PER_MINUTE` | `30` | Per-IP limit on `/api/chat` (0 disables). |
| `ADMIN_TOKEN` | — | If set, required as `X-Admin-Token` for `POST /api/reindex`. |
| `FALLBACK_CONTACT` | — | e.g. `email sales@example.com` — appended to fallback answers. |

Secrets stay server-side: the widget only ever calls `/api/chat` on your
backend; the LLM key is never exposed to the browser.

## 3. The product CSV

`data/products.csv` — one row per product:

| Column | Required | Example / notes |
|---|---|---|
| `sku` | ✅ | `AUR-H700` — must be unique |
| `name` | ✅ | `Aurora Wireless Headphones` |
| `price` | ✅ | `129.99` (currency symbols are stripped) |
| `description` | ✅ | 1–2 sentences; this is what the LLM quotes |
| `category` | | `Audio` |
| `currency` | | `USD` (default) |
| `availability` | | `In stock`, `Backordered`, … |
| `specs` | | `Battery life: 30h; Bluetooth: 5.3; Weight: 250g` |
| `tags` | | `headphones; wireless; anc` — extra search keywords |
| `url` | | Product page link → shown on the widget's card |

The `specs` format (`Key: Value; Key: Value`) deliberately keeps **one CSV
format for any product type** — laptops, chairs and cameras can all have
different specs without new columns.

**Updating products requires no code changes:** edit/save the CSV and the
backend re-reads and re-indexes it automatically on the next request
(mtime-based hot reload). Validation warnings (missing price, duplicate SKU)
go to the log and never crash the service. Force a refresh with:

```bash
curl -X POST http://localhost:8000/api/reindex -H "X-Admin-Token: $ADMIN_TOKEN"
```

## 4. Embedding the widget

One script tag on any HTML page:

```html
<script src="https://YOUR-SERVER/widget.js"
        data-title="Store Assistant"
        data-subtitle="Ask about products, specs & pricing"
        data-color="#4f46e5"
        data-position="right"
        async></script>
```

| Attribute | Default | |
|---|---|---|
| `data-api-url` | same origin as `widget.js` | Backend URL if hosted elsewhere (CORS must allow your site) |
| `data-title` / `data-subtitle` | Product Assistant / Ask about… | Header text |
| `data-color` | `#4f46e5` | Accent color |
| `data-position` | `right` | `left` or `right` |
| `data-suggestions` | sample questions | Semicolon-separated starter chips |

The widget includes the launcher, chat window, typing indicator, product
cards (SKU · price · availability), error states, Enter-to-send,
Escape-to-close, and a full-screen layout on phones (with safe-area support).
No dependencies; styles are injected automatically.

## 5. How answers stay grounded (anti-hallucination)

Four independent layers:

1. **Retrieval threshold** — the question is vector-searched against the
   catalog; if the best match scores below `RELEVANCE_THRESHOLD`, the bot
   returns a friendly fallback *without calling the LLM at all*. Off-topic
   questions ("capital of france?") never reach the model.
2. **Verbatim context** — the LLM prompt contains only the top-k raw CSV rows
   plus a strict system prompt: use only this context; if the answer isn't in
   it, say so; quote prices exactly; never invent SKUs or specs.
3. **SKU guard** — any product code in the generated answer that doesn't
   exist in the catalog triggers a visible caution note.
4. **Graceful degradation** — if the LLM is down or unconfigured, the bot
   returns the matched products directly instead of guessing.

The retrieval layer is keyword-based (TF-IDF over words + character n-grams,
with light stemming and stopword removal). That is ideal for catalog facts
(product names, SKUs, "waterproof", "4k"), and it means *every* retrieved row
is genuinely relevant — the LLM's job is reading and phrasing, not finding.

## 6. Testing

```bash
.venv/bin/python -m pytest
```

29 tests, no network or API key required:

- `tests/test_retriever.py` — **accuracy**: 25 real customer questions must
  retrieve the correct product family above the threshold (price, spec,
  SKU, comparison and typo queries); 9 off-topic questions must fall below
  it; CSV hot-reload test.
- `tests/test_rag_fallback.py` — **fallbacks**: off-topic → fallback with
  zero LLM calls; hallucinated-SKU caution note; LLM-outage degradation;
  no-LLM mode; follow-up retrieval via history.
- `tests/test_api.py` — endpoints, validation, rate limiting, admin token,
  CORS, widget/demo serving, broken-CSV safety.

When you swap in your own CSV, run the accuracy tests and adjust
`ACCURACY_CASES` (in `tests/test_retriever.py`) to your products — it doubles
as a regression check for your catalog data.

## 7. Deployment

### Option A — Docker (recommended)

```bash
cp .env.example .env          # set LLM_API_KEY, CORS_ORIGINS, ADMIN_TOKEN
docker compose up -d --build
```

The `data/` directory is bind-mounted, so CSV edits apply instantly without
rebuilding.

### Option B — systemd + uvicorn

```bash
sudo rsync -a --exclude .venv ./ /opt/rag_chatbot/
cd /opt/rag_chatbot && sudo python3 -m venv .venv
sudo .venv/bin/pip install -r requirements.txt
sudo cp .env.example /opt/rag_chatbot/.env && sudo nano /opt/rag_chatbot/.env
sudo cp deploy/rag-chatbot.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now rag-chatbot
```

### Behind HTTPS (required in production)

Put a reverse proxy in front — e.g. Caddy (automatic TLS):

```
yourstore.com {
    reverse_proxy localhost:8000
}
```

or nginx with certbot. Then set `CORS_ORIGINS` to your real site origin(s)
instead of `*`, and serve your site over the same domain to keep everything
first-party.

### Production checklist

- [ ] `LLM_API_KEY` set in `.env`, file permissions `chmod 600 .env`
- [ ] `CORS_ORIGINS` restricted to your site(s)
- [ ] `ADMIN_TOKEN` set (protects `/api/reindex`)
- [ ] HTTPS via reverse proxy
- [ ] `RATE_LIMIT_PER_MINUTE` appropriate (default 30/min per IP)
- [ ] Log monitoring (`journalctl -u rag-chatbot` or `docker logs`)

## 8. Maintenance

| Task | How |
|---|---|
| Add/edit products | Edit `data/products.csv` — live within one request |
| Force re-index | `POST /api/reindex` (with `X-Admin-Token`) |
| Change LLM provider | Update `LLM_BASE_URL` + `LLM_MODEL` + key, restart |
| Better semantic recall | `RETRIEVER=openai` (uses the embeddings API; costs pennies at this scale) |
| Too many false fallbacks | Lower `RELEVANCE_THRESHOLD` (e.g. `0.07`) |
| Too many off-topic answers | Raise it (e.g. `0.15`) and/or review the LLM prompt in `backend/rag.py` |
| Re-check answer quality | `pytest` + a few manual questions via `/` demo page |

### Scaling notes

At ~50 SKUs, one uvicorn worker handles well beyond typical traffic
(indexing is in-memory; each search is <1ms). If you grow to thousands of
products or need multi-instance deployment, switch `RETRIEVER=openai` and
put a shared cache in front — the code path is already isolated in
`backend/retriever.py`. The rate limiter is per-process; for multiple
workers/hosts, use the reverse proxy's rate limiting instead.

## 9. API reference

| Endpoint | Description |
|---|---|
| `POST /api/chat` | Body `{"message": "...", "history": [{"role": "user", "content": "..."}]}` → `{"answer", "fallback", "products": [...]}` |
| `GET /api/products` | Full catalog as JSON |
| `POST /api/reindex` | Reload CSV + rebuild index (optional `X-Admin-Token`) |
| `GET /health` | Status, product count, LLM config |
| `GET /widget.js` · `GET /` | Widget script · demo page |

## 10. Troubleshooting

| Symptom | Fix |
|---|---|
| Bot always replies with product lists, no prose | No `LLM_API_KEY` set (that's no-LLM mode) — check `.env` and restart |
| 429 responses | Rate limit hit — raise `RATE_LIMIT_PER_MINUTE` or wait |
| 402/401 from the LLM | Invalid key or wrong `LLM_BASE_URL` for your provider |
| Widget doesn't open on your site | Check the browser console: wrong `data-api-url`, or CORS not allowing your origin |
| CSV edits not appearing | Log will show reload warnings — check required columns and unique SKUs; `POST /api/reindex` to force |
| Wrong products retrieved | Add `tags` keywords to those rows; check `description` mentions the feature customers ask about |

## 11. Project structure

```
backend/
  main.py       FastAPI app, routes, rate limiting, admin auth
  rag.py        Pipeline: retrieve → threshold → prompt → generate → SKU guard
  retriever.py  TF-IDF index (default) + optional OpenAI embeddings index
  catalog.py    CSV loading, validation, hot reload
  llm.py        OpenAI-compatible chat client
  config.py     .env / environment settings
  models.py     Pydantic schemas
data/products.csv      54 sample products — replace with your catalog
static/chat-widget.js  Self-contained widget (embedded CSS, no deps)
static/demo.html       Demo storefront embedding the widget
tests/                 Accuracy, fallback and API tests (29 tests)
deploy/                systemd unit
Dockerfile, docker-compose.yml
```
