"""Anti-hallucination and fallback behavior of the RAG pipeline."""
from __future__ import annotations

import pytest

from backend.models import ChatRequest, ChatTurn
from backend.rag import RagPipeline

from conftest import FakeLLM, NoLLM


def make_pipeline(index, catalog, llm, settings) -> RagPipeline:
    return RagPipeline(index=index, catalog=catalog, llm=llm, settings=settings)


async def test_off_topic_gets_fallback_without_llm_call(index, catalog, settings):
    """Off-topic questions must never reach the LLM."""
    llm = FakeLLM()
    pipeline = make_pipeline(index, catalog, llm, settings)

    resp = await pipeline.answer(ChatRequest(message="best pizza near me"))

    assert resp.fallback is True
    assert resp.products == []
    assert "couldn't find anything matching" in resp.answer
    assert llm.calls == [], "LLM must not be called for off-topic questions"


async def test_in_domain_question_uses_catalog_data_only(index, catalog, settings):
    llm = FakeLLM(reply="The AUR-H700 costs $129.99 and lasts 30 hours.")
    pipeline = make_pipeline(index, catalog, llm, settings)

    resp = await pipeline.answer(
        ChatRequest(message="how much do the Aurora Wireless Headphones cost?")
    )

    assert resp.fallback is False
    assert llm.calls, "LLM should be called for in-domain questions"
    prompt = llm.calls[0]["user"]
    # The prompt must contain verbatim catalog data for the matched product.
    assert "AUR-H700" in prompt
    assert "129.99" in prompt
    assert "Battery life: 30h" in prompt
    assert "CUSTOMER QUESTION:" in prompt
    # Matched products are returned for widget cards.
    assert any(p.sku == "AUR-H700" for p in resp.products)


async def test_prompt_restricts_to_context(index, catalog, settings):
    """System prompt must demand answers grounded only in the catalog."""
    llm = FakeLLM()
    pipeline = make_pipeline(index, catalog, llm, settings)
    await pipeline.answer(ChatRequest(message="tell me about the Kronos mouse"))

    system = llm.calls[0]["system"]
    assert "ONLY" in system and "CATALOG CONTEXT" in system
    assert "not in the catalog" in system


async def test_sku_guard_flags_hallucinated_product_code(index, catalog, settings):
    """If the LLM invents a SKU, a caution note is appended to the answer."""
    llm = FakeLLM(reply="You'll love the AUR-H999 Pro Max (AUR-H999).")
    pipeline = make_pipeline(index, catalog, llm, settings)

    resp = await pipeline.answer(ChatRequest(message="aurora headphones"))

    assert "double-check" in resp.answer
    assert "AUR-H999" in resp.answer  # original text preserved


async def test_sku_guard_passes_answers_with_real_skus(index, catalog, settings):
    llm = FakeLLM(reply="The AUR-H700 costs 129.99 USD and the AUR-H900 costs 249.99 USD.")
    pipeline = make_pipeline(index, catalog, llm, settings)

    resp = await pipeline.answer(ChatRequest(message="aurora headphones price"))

    assert "double-check" not in resp.answer


async def test_llm_outage_degrades_gracefully(index, catalog, settings):
    """LLM errors must never surface as 500s — we list matched products."""
    llm = FakeLLM(fail=True)
    pipeline = make_pipeline(index, catalog, llm, settings)

    resp = await pipeline.answer(ChatRequest(message="waterproof speaker"))

    assert resp.fallback is False
    assert "VOL-S30" in resp.answer
    assert any(p.sku == "VOL-S30" for p in resp.products)


async def test_no_llm_mode_answers_from_catalog(index, catalog, settings):
    pipeline = make_pipeline(index, catalog, NoLLM(), settings)

    resp = await pipeline.answer(ChatRequest(message="how much is the smart plug?"))

    assert resp.fallback is False
    assert "HOM-P3" in resp.answer
    assert "29.99" in resp.answer
    assert resp.products


async def test_follow_up_question_uses_history_for_retrieval(index, catalog, settings):
    """A vague follow-up should still retrieve the right products via history."""
    llm = FakeLLM(reply="The cheaper Lite model costs 79.99 USD.")
    pipeline = make_pipeline(index, catalog, llm, settings)

    request = ChatRequest(
        message="what about the cheaper one?",
        history=[
            ChatTurn(role="user", content="tell me about the Aurora headphones"),
            ChatTurn(role="assistant", content="The AUR-H700 costs 129.99 USD."),
        ],
    )
    resp = await pipeline.answer(request)

    assert resp.fallback is False
    assert llm.calls
    assert "CONVERSATION SO FAR" in llm.calls[0]["user"]


async def test_follow_up_history_wins_over_threshold_on_wrong_products(index, catalog, settings):
    """Regression: a follow-up containing a catalog term ("multipoint") can
    score above threshold on *unrelated* products, which used to skip the
    history retry entirely. History must still be used."""
    llm = FakeLLM(reply="The AUR-H700 does not list multipoint; the AUR-H900 does.")
    pipeline = make_pipeline(index, catalog, llm, settings)

    request = ChatRequest(
        message="and does the cheaper one support multipoint bluetooth?",
        history=[
            ChatTurn(role="user", content="difference between AUR-H700 and AUR-H900"),
            ChatTurn(
                role="assistant",
                content="The AUR-H900 costs 249.99 USD, the AUR-H700 costs 129.99 USD.",
            ),
        ],
    )
    resp = await pipeline.answer(request)

    skus = [p.sku for p in resp.products]
    assert "AUR-H700" in skus or "AUR-H900" in skus, (
        f"follow-up should retrieve the discussed products, got {skus}"
    )
    assert "CONVERSATION SO FAR" in llm.calls[0]["user"]


async def test_empty_catalog_answers_are_safe(index, catalog, settings, tmp_path):
    """Even with a tiny catalog, unrelated questions must fall back."""
    from backend.catalog import Catalog
    from backend.retriever import TfidfIndex

    tiny = tmp_path / "tiny.csv"
    tiny.write_text(
        "sku,name,category,price,currency,availability,description,specs,tags\n"
        "TST-1,Test Lamp,Office,10.00,USD,In stock,A simple test lamp.,Wattage: 10W,lamp\n"
    )
    pipeline = make_pipeline(TfidfIndex(Catalog(tiny)), Catalog(tiny), FakeLLM(), settings)

    resp = await pipeline.answer(ChatRequest(message="do you sell cars?"))
    assert resp.fallback is True


@pytest.mark.parametrize(
    "message",
    ["", "a" * 1200],
)
async def test_request_validation_rejects_bad_messages(index, catalog, settings, message):
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ChatRequest(message=message)
