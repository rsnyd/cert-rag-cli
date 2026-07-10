"""Query embedding via Voyage."""
import voyageai

EMBED_MODEL = "voyage-3-lite"

def embed_query(query: str) -> list[float]:
    voyage = voyageai.Client()
    return voyage.embed([query], model=EMBED_MODEL, input_type="query").embeddings[0]