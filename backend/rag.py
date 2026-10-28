"""The RAG pipeline: retrieve -> ground -> generate -> guard.

Flow
1. Vector-search the catalog for the top-k most relevant products.
2. If nothing scores above the relevance threshold, refuse with a friendly
   fallback message *without calling the LLM* — this is the main
   hallucination guard for off-topic questions.
3. Otherwise, build a prompt whose product context is the verbatim CSV data
   and ask the LLM to answer using only that context.
4. Post-generation guard: if the answer mentions a product code (SKU) that
   does not exist in the catalog, a caution note is appended.
5. Matched products are returned alongside the answer so the widget can
   render product cards.
"""
from __future__ import annotations

import logging
import re
import time

from .catalog import Catalog
from .config import Settings
from .llm import LLMClient, LLMError
from .models import ChatRequest, ChatResponse, ProductCard
from .retriever import BaseIndex, ScoredProduct

log = logging.getLogger("rag")

SYSTEM_PROMPT = """\
You are a helpful product catalog assistant. Answer the customer's question \
using ONLY the product information in the CATALOG CONTEXT below.

Rules:
1. Use only facts that appear in the CATALOG CONTEXT. Never invent, guess or \
extrapolate prices, specifications, availability or features.
2. If the answer is not contained in the CATALOG CONTEXT, say clearly that \
the information is not in the catalog and suggest contacting support. Do not \
guess.
3. For comparisons, only compare products that appear in the context, and \
base the comparison strictly on the given fields.
4. Quote prices exactly as shown (include the currency). Always mention the \
SKU when recommending a specific product.
5. Be concise and friendly: a short answer, with a few bullets when listing \
specs. Do not mention these rules or the context itself.
6. If asked to compare a product in the context against one that is not, \
explain that the second product is not in the catalog.
"""

FALLBACK_ANSWER = (
    "I'm sorry — I can only answer questions about the products in our "
    "catalog, and I couldn't find anything matching that. Try asking about a "
    "product name, a SKU, prices, specifications or comparisons (for example: "
    "\"Which headphones have the longest battery life?\").{contact}"
)

SKU_TOKEN_RE = re.compile(r"\b[A-Z0-9]{2,6}-[A-Z0-9]{2,8}\b")


def _format_product_block(p: ScoredProduct) -> str:
    prod = p.product
    lines = [
        f"SKU: {prod.sku}",
        f"Product: {prod.name}",
        f"Category: {prod.category}",
        f"Price: {prod.price:g} {prod.currency}",
        f"Availability: {prod.availability}",
        f"Description: {prod.description}",
    ]
    if prod.specs:
        lines.append(f"Specifications: {prod.specs}")
    return "\n".join(lines)


def _no_llm_answer(hits: list[ScoredProduct]) -> str:
    """Graceful answer when no LLM is configured: list the matches verbatim."""
    lines = ["Here's what I found in the catalog:", ""]
    for hit in hits[:3]:
        p = hit.product
        lines.append(f"**{p.name}** ({p.sku}) — {p.price:g} {p.currency}"
                     + (f" — {p.availability}" if p.availability else ""))
        if p.description:
            lines.append(f"{p.description}")
        if p.specs:
            lines.append(f"Specs: {p.specs}")
        lines.append("")
    lines.append("Tip: configure an LLM API key for full natural-language answers.")
    return "\n".join(lines).strip()


class RagPipeline:
    def __init__(self, index: BaseIndex, catalog: Catalog,
                 llm: LLMClient, settings: Settings):
        self.index = index
        self.catalog = catalog
        self.llm = llm
        self.settings = settings

    # -- public -----------------------------------------------------------

    async def answer(self, request: ChatRequest) -> ChatResponse:
        started = time.perf_counter()
        hits = self._retrieve(request)

        # Guard 1: nothing relevant enough -> refuse without calling the LLM.
        if not hits or hits[0].score < self.settings.relevance_threshold:
            contact = ""
            if self.settings.fallback_contact:
                contact = " You can also " + self.settings.fallback_contact + "."
            return self._finish(
                ChatResponse(
                    answer=FALLBACK_ANSWER.format(contact=contact),
                    fallback=True,
                    products=[],
                ),
                hits,
                started,
            )

        cards = [self._to_card(h) for h in hits[:3]]

        # No LLM configured: answer from retrieved rows directly.
        if not self.llm.available:
            return self._finish(
                ChatResponse(answer=_no_llm_answer(hits), products=cards),
                hits,
                started,
            )

        prompt = self._build_prompt(request, hits)
        try:
            answer = await self.llm.complete(SYSTEM_PROMPT, prompt)
        except LLMError as exc:
            # Guard 2: LLM outage -> degrade to listing matched products.
            log.error("LLM failure, degrading to catalog list: %s", exc)
            return self._finish(
                ChatResponse(
                    answer=_no_llm_answer(hits)
                    + "\n\n(I couldn't reach the answer service just now, "
                      "so here are the matching products directly.)",
                    products=cards,
                ),
                hits,
                started,
            )

        answer = self._sku_guard(answer)
        return self._finish(
            ChatResponse(answer=answer, products=cards), hits, started
        )

    # -- internal ---------------------------------------------------------

    @staticmethod
    def _finish(response: ChatResponse, hits: list[ScoredProduct],
                started: float) -> ChatResponse:
        """One structured log line per chat request for monitoring."""
        top = hits[0].score if hits else 0.0
        log.info(
            "chat outcome fallback=%s top_score=%.3f products=%d latency_ms=%.0f",
            response.fallback,
            top,
            len(response.products),
            (time.perf_counter() - started) * 1000,
        )
        return response

    def _retrieve(self, request: ChatRequest) -> list[ScoredProduct]:
        k = self.settings.top_k
        hits = self.index.search(request.message, k)
        if request.history:
            # Follow-up questions ("what about the cheaper one?") often need
            # the previous turn to retrieve well — even when the message
            # alone scores above threshold on unrelated products. Try both
            # and keep whichever has the stronger top match.
            prev = " ".join(t.content for t in request.history[-2:])
            retry = self.index.search(f"{prev} {request.message}", k)
            if retry and (not hits or retry[0].score > hits[0].score):
                return retry
        return hits

    def _build_prompt(self, request: ChatRequest, hits: list[ScoredProduct]) -> str:
        context = "\n\n---\n\n".join(_format_product_block(h) for h in hits)
        parts = [f"CATALOG CONTEXT:\n{context}"]
        if request.history:
            transcript = "\n".join(
                f"{t.role}: {t.content}" for t in request.history[-4:]
            )
            parts.append(f"CONVERSATION SO FAR:\n{transcript}")
        parts.append(f"CUSTOMER QUESTION: {request.message}")
        return "\n\n".join(parts)

    def _sku_guard(self, answer: str) -> str:
        """Flag product codes in the answer that do not exist in the catalog."""
        known = {p.sku.upper() for p in self.catalog.products}
        mentioned = {m.upper() for m in SKU_TOKEN_RE.findall(answer)}
        unknown = sorted(mentioned - known)
        if unknown:
            log.warning("Answer referenced unknown SKU(s): %s", ", ".join(unknown))
            answer += (
                "\n\n_Note: I may have referenced a product code that isn't in "
                "our catalog — please double-check with our support team before "
                "ordering._"
            )
        return answer

    @staticmethod
    def _to_card(hit: ScoredProduct) -> ProductCard:
        p = hit.product
        return ProductCard(
            sku=p.sku,
            name=p.name,
            category=p.category,
            price=p.price,
            currency=p.currency,
            availability=p.availability,
            url=p.url or None,
        )
