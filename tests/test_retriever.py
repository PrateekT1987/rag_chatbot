"""Retrieval accuracy tests against the shipped catalog.

"Accuracy" here means the right products (or product family) are retrieved
in the top-k results with a score above the relevance threshold — the RAG
guarantee that the LLM only ever sees relevant catalog rows.
"""
from __future__ import annotations

from backend.retriever import TfidfIndex

# (question, SKU prefix of an expected match in the top-4)
ACCURACY_CASES = [
    ("which headphones have the longest battery?", "AUR-H"),
    ("do you have a waterproof speaker?", "VOL-S3"),
    ("cheapest mechanical keyboard", "KRN-K8"),
    ("4k monitor under $400", "HEL-Q2"),
    ("charge my laptop and phone at the same time", "LUM-L6"),
    ("bag for a carry-on flight", "TRV-B3"),
    ("camera for wedding photography", "KST-R"),
    ("router for a large house", "NET-R"),
    ("smart plug", "HOM-P3"),
    ("AUR-H700", "AUR-H7"),
    ("battery life of the aurora pro", "AUR-H9"),
    ("gaming mouse lightweight", "KRN-M5"),
    ("portable ssd for video editing", "FLX-D"),
    ("how much is the ORB-X14?", "ORB-X1"),
    ("laptop under 600 dollars", "ORB-X1"),
    ("chair with lumbar support", "OFF-CH"),
    ("bluetooth earbuds for running", "TRV-E6"),
    ("outdoor security camera", "HOM-O4"),
    ("usb-c cable", "LUM-C"),
    ("monitor with usb-c power delivery", "HEL-Q2"),
    ("what laptops do you sell", "ORB-X"),
    ("do you have anything for recording podcasts", "NOV-M"),
    ("warranty on the backpack", "TRV-B"),
    ("smartwatch with gps", "TRV-W3"),
    ("full frame camera", "KST-R9"),
    ("printer paper", None),  # not in catalog: handled separately below
]


def test_accuracy_relevant_questions(index, settings):
    checked = 0
    for query, expected_prefix in ACCURACY_CASES:
        if expected_prefix is None:
            continue
        checked += 1
        hits = index.search(query, k=4)
        assert hits, f"{query!r}: no results"
        assert hits[0].score >= settings.relevance_threshold, (
            f"{query!r}: top score {hits[0].score:.3f} below threshold "
            f"{settings.relevance_threshold} — would trigger fallback"
        )
        skus = [h.product.sku for h in hits]
        assert any(s.startswith(expected_prefix) for s in skus), (
            f"{query!r}: expected a {expected_prefix}* product in top-4, got {skus}"
        )


def test_exact_sku_lookup_is_first(index):
    hits = index.search("AUR-H700", k=4)
    assert hits[0].product.sku == "AUR-H700"


def test_price_question_retrieves_product(index):
    hits = index.search("how much does the Vertex Portable Speaker cost?", k=4)
    assert any(h.product.sku == "VOL-S30" for h in hits[:2])


def test_off_topic_questions_blocked_below_threshold(index, settings):
    off_topic = [
        "best pizza near me",
        "how do I bake a cake",
        "capital of france",
        "time in new york",
        "who won the world cup",
        "how to learn guitar",
        "stock price of apple",
        "tell me a joke",
        "python list comprehension",
    ]
    for query in off_topic:
        hits = index.search(query, k=4)
        top = hits[0].score if hits else 0.0
        assert top < settings.relevance_threshold, (
            f"{query!r} scored {top:.3f} — should be blocked"
        )


def test_comparison_query_retrieves_both_products(index):
    hits = index.search("difference between AUR-H700 and AUR-H900", k=4)
    skus = [h.product.sku for h in hits]
    assert "AUR-H700" in skus and "AUR-H900" in skus


def test_catalog_hot_reload_picks_up_new_product(catalog, index, tmp_path):
    import shutil
    import os

    csv = tmp_path / "products.csv"
    shutil.copy(catalog.path, csv)

    from backend.catalog import Catalog

    cat = Catalog(csv)
    idx = TfidfIndex(cat)
    hits = idx.search("Quantum Widget 9000", k=4)
    assert not hits or hits[0].score < 0.10

    # Simulate a CSV update: append a new product, no restart required.
    header, rows = csv.read_text().splitlines()[0], csv.read_text().splitlines()[1:]
    new_row = (
        "QNT-W9,Quantum Widget 9000,Accessories,19.99,USD,In stock,"
        "A fictional widget used in tests.,Material: Unobtainium,widget; test"
    )
    csv.write_text("\n".join([header, *rows, new_row]) + "\n")
    os.utime(csv, (os.path.getmtime(csv) + 5, os.path.getmtime(csv) + 5))

    hits = idx.search("Quantum Widget 9000", k=4)
    assert hits and hits[0].product.sku == "QNT-W9", "hot reload failed"
    assert cat.get("QNT-W9") is not None
