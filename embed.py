"""Day 11: Embed chunks and store in a local Chroma collection."""
import json
import os
import time
from pathlib import Path

import chromadb
import voyageai
from rich.progress import track

CHUNKS_FILE = Path("data/chunks.jsonl")
CHROMA_DIR = Path(".chroma")
COLLECTION_NAME = "sqf_docs"
EMBED_MODEL = "voyage-3-lite"   # cheap, good enough for learning; upgrade to voyage-3 later


# Chroma metadata values must be str/int/float/bool - never None. Drop None keys.
def _clean_meta(rec: dict) -> dict:
    fields = ("source", "doc_type", "edition", "module", "clause", "clause_source",
              "clause_title", "page")
    return {k: rec[k] for k in fields if rec.get(k) is not None}


def main():
    voyage = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])
    chroma = chromadb.PersistentClient(path=str(CHROMA_DIR))

    try:
        chroma.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = chroma.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    records = [json.loads(line) for line in CHUNKS_FILE.open()]
    print(f"Embedding {len(records)} chunks...")

    # Free tier: 3 RPM, 10K TPM -> small batches + sleep.
    BATCH = 8
    SLEEP_BETWEEN_BATCHES = 21
    batches = list(range(0, len(records), BATCH))
    for idx, i in enumerate(track(batches, description="Embedding")):
        batch = records[i:i + BATCH]
        texts = [r["text"] for r in batch]

        result = voyage.embed(texts=texts, model=EMBED_MODEL, input_type="document")

        collection.add(
            ids=[r["id"] for r in batch],
            embeddings=result.embeddings,
            documents=texts,
            metadatas=[_clean_meta(r) for r in batch],
        )
        if idx < len(batches) - 1:
            time.sleep(SLEEP_BETWEEN_BATCHES)

    print(f"Stored {collection.count()} vectors in collection '{COLLECTION_NAME}'.")


if __name__ == "__main__":
    main()