#!/usr/bin/env python3
"""
RAG Retrieval Evaluation — Language Tutor Agent.

Runs sample queries against Pinecone and checks that retrieved results
match expected topics. This is a lightweight evaluation script — no
framework dependencies, just assert-based checks.

Usage:
    # Make sure .env is configured and Pinecone is set up:
    python -m app.pinecone_setup
    # Then run:
    python tests/test_rag_eval.py
"""

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from pinecone import Pinecone

# Add backend to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.tools import init_vector_store

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

INDEX_NAME = os.getenv("PINECONE_INDEX", "language-tutor")

# Sample queries per language with expected topics
QUERIES = {
    "en": [
        ("past tense verbs", ["tenses", "past", "verbs", "grammar"]),
        ("ordering food at restaurant", ["food", "restaurant", "greetings", "vocabulary", "vocab"]),
        ("present progressive grammar", ["tenses", "continuous", "grammar", "progressive"]),
        ("daily routine vocabulary", ["daily_routine", "vocabulary", "vocab", "routine"]),
        ("travel phrases", ["travel", "greetings", "vocabulary", "vocab"]),
    ],
    "ko": [
        ("한국어 과거 시제", ["tenses", "past", "grammar"]),
        ("식당에서 주문하기", ["food", "greetings", "vocabulary", "vocab"]),
        ("일상 표현", ["daily_routine", "vocabulary", "vocab"]),
        ("여행 한국어", ["travel", "greetings", "vocabulary", "vocab"]),
        ("감정 표현", ["emotions", "adjectives", "vocabulary", "vocab"]),
    ],
    "ja": [
        ("日本語の過去形", ["tenses", "past", "grammar"]),
        ("レストランでの会話", ["food", "greetings", "vocabulary", "vocab"]),
        ("日常会話の練習", ["daily_routine", "vocabulary", "vocab"]),
        ("旅行の日本語", ["travel", "greetings", "vocabulary", "vocab"]),
        ("感情を表す言葉", ["emotions", "adjectives", "vocabulary", "vocab"]),
    ],
}


def check_topic_match(retrieved_topics: list[str], expected_topics: list[str]) -> bool:
    """Check if any retrieved topic matches any expected topic (case-insensitive)."""
    retrieved_lower = [t.lower() for t in retrieved_topics]
    for expected in expected_topics:
        expected_lower = expected.lower()
        for retrieved in retrieved_lower:
            if expected_lower in retrieved or retrieved in expected_lower:
                return True
    return False


def main():
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    gemini_api_key = os.getenv("GEMINI_API_KEY")

    if not pinecone_api_key or not gemini_api_key:
        print("ERROR: PINECONE_API_KEY and GEMINI_API_KEY must be set.")
        sys.exit(1)

    pc = Pinecone(api_key=pinecone_api_key)

    try:
        index = pc.Index(INDEX_NAME)
    except Exception:
        print(f"ERROR: Could not connect to Pinecone index '{INDEX_NAME}'")
        print("Run: python -m app.pinecone_setup")
        sys.exit(1)

    embedding_api_key = os.getenv("GOOGLE_EMBEDDING_API_KEY") or gemini_api_key
    init_vector_store(index, embedding_api_key)

    from app.tools import _get_retriever

    print("=" * 60)
    print("RAG Retrieval Evaluation — Language Tutor Agent")
    print("=" * 60)
    print()

    total = 0
    passed = 0

    for language, queries in QUERIES.items():
        print(f"--- {language} ---")
        for query_text, expected_topics in queries:
            total += 1

            try:
                retriever = _get_retriever(language, k=5)
                docs = retriever.invoke(query_text)

                if not docs:
                    print(f"  ✗ FAIL: '{query_text}' — No results retrieved")
                    continue

                retrieved_topics = []
                for doc in docs:
                    topic = doc.metadata.get("topic", "")
                    title = doc.metadata.get("title", "")
                    if topic:
                        retrieved_topics.append(topic)
                    if title:
                        retrieved_topics.append(title)

                matched = check_topic_match(retrieved_topics, expected_topics)

                if matched:
                    print(f"  ✓ PASS: '{query_text}' → {retrieved_topics[:3]}")
                    passed += 1
                else:
                    print(f"  ✗ FAIL: '{query_text}' → {retrieved_topics[:3]} (expected topics: {expected_topics})")

            except Exception as exc:
                print(f"  ✗ ERROR: '{query_text}' — {exc}")

        print()

    print("-" * 60)
    print(f"Results: {passed}/{total} queries matched expected topics")
    print(f"Score:  {(passed / total * 100):.1f}%" if total > 0 else "Score: N/A")
    print("-" * 60)

    # Threshold: 70% minimum pass rate
    if total > 0 and (passed / total) < 0.70:
        print("\n⚠  Retrieval quality below 70% threshold — consider reviewing seed data.")
        sys.exit(1)
    else:
        print("\n✓ Retrieval quality meets threshold.")
        sys.exit(0)


if __name__ == "__main__":
    main()