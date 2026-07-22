"""
Pinecone index creation, embedding, and upsert for the Language Tutor Agent.

Usage:
    python -m app.pinecone_setup          # create index and upsert all seed data
    python -m app.pinecone_setup --reset  # delete and recreate index before upsert
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google import genai
from pinecone import Pinecone, ServerlessSpec

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

INDEX_NAME = os.getenv("PINECONE_INDEX", "language-tutor")
EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIMENSIONS = 3072
NAMESPACES = ["en", "ko", "ja"]

SEED_FILES = [
    ("en", "grammar", "seed_grammar_en.json"),
    ("en", "vocab", "seed_vocab_en.json"),
    ("ko", "grammar", "seed_grammar_ko.json"),
    ("ko", "vocab", "seed_vocab_ko.json"),
    ("ja", "grammar", "seed_grammar_ja.json"),
    ("ja", "vocab", "seed_vocab_ja.json"),
]


def load_seed_data(language: str, content_type: str, filename: str) -> list[dict[str, Any]]:
    """Load seed data from a JSON file."""
    path = DATA_DIR / filename
    if not path.exists():
        print(f"[WARN] Seed file not found: {path}")
        return []
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"  Loaded {len(data)} {content_type} entries for {language} from {filename}")
    return data


def create_index(pc: Pinecone, reset: bool = False) -> None:
    """Create the Pinecone index if it doesn't exist."""
    existing = pc.list_indexes()
    existing_names = [idx.name if hasattr(idx, "name") else idx.get("name", "") for idx in existing]

    if INDEX_NAME in existing_names:
        if reset:
            print(f"Deleting existing index '{INDEX_NAME}'...")
            pc.delete_index(INDEX_NAME)
        else:
            print(f"Index '{INDEX_NAME}' already exists. Use --reset to recreate.")
            return

    print(f"Creating serverless index '{INDEX_NAME}' (dim={EMBEDDING_DIMENSIONS}, cosine)...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=EMBEDDING_DIMENSIONS,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    print("Index created successfully.")


def embed_texts(client: genai.Client, texts: list[str]) -> list[list[float]]:
    """Generate embeddings for a batch of texts using Google GenAI."""
    embeddings = []
    for text in texts:
        result = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=text,
        )
        if result.embeddings:
            embeddings.append(result.embeddings[0].values)
        else:
            # Fallback: zero vector (shouldn't happen in practice)
            embeddings.append([0.0] * EMBEDDING_DIMENSIONS)
    return embeddings


def upsert_seed_data(pc: Pinecone, genai_client: genai.Client) -> None:
    """Embed and upsert all seed data to Pinecone."""
    index = pc.Index(INDEX_NAME)

    total_upserted = 0

    for language, content_type, filename in SEED_FILES:
        entries = load_seed_data(language, content_type, filename)
        if not entries:
            continue

        print(f"Embedding {len(entries)} {language}/{content_type} entries...")
        texts = [entry["content"] for entry in entries]
        embeddings = embed_texts(genai_client, texts)

        vectors = []
        for entry, embedding in zip(entries, embeddings):
            vectors.append({
                "id": entry["id"],
                "values": embedding,
                "metadata": {**entry["metadata"], "text": entry["content"]},
            })

        # Upsert in batches of 100
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i : i + batch_size]
            index.upsert(vectors=batch, namespace=language)

        total_upserted += len(vectors)
        print(f"  Upserted {len(vectors)} vectors to namespace '{language}'")

    print(f"\nTotal upserted: {total_upserted} vectors across {len(NAMESPACES)} namespaces.")


def main() -> None:
    load_dotenv(BASE_DIR / ".env")

    parser = argparse.ArgumentParser(description="Pinecone setup for Language Tutor Agent")
    parser.add_argument("--reset", action="store_true", help="Delete and recreate the index")
    args = parser.parse_args()

    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    gemini_api_key = os.getenv("GEMINI_API_KEY")

    if not pinecone_api_key:
        raise RuntimeError("PINECONE_API_KEY environment variable is required")
    if not gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable is required")

    pc = Pinecone(api_key=pinecone_api_key)
    genai_client = genai.Client(api_key=gemini_api_key)

    create_index(pc, reset=args.reset)
    upsert_seed_data(pc, genai_client)

    print("\nPinecone setup complete.")


if __name__ == "__main__":
    main()