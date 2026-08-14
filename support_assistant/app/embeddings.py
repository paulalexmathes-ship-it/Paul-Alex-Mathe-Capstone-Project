
"""Document loading, embedding, and ChromaDB indexing."""
import os
import chromadb
from sentence_transformers import SentenceTransformer
from app.config import CHROMA_COLLECTION_NAME, CHROMA_PERSIST_DIR, EMBEDDING_MODEL, DOCS_DIR


def load_documents() -> list[dict]:
    """Load all documents from the docs directory."""
    documents = []
    for filename in sorted(os.listdir(DOCS_DIR)):
        if filename.endswith(".txt"):
            filepath = os.path.join(DOCS_DIR, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read().strip()
            doc_id = filename.replace(".txt", "")  # e.g., "doc_01"
            documents.append({
                "id": doc_id,
                "content": content,
                "filename": filename
            })
    return documents


def get_embedding_model() -> SentenceTransformer:
    """Load the sentence-transformers embedding model."""
    return SentenceTransformer(EMBEDDING_MODEL)


def initialize_chromadb(documents: list[dict], model: SentenceTransformer) -> chromadb.Collection:
    """Embed documents and store in ChromaDB collection."""
    client = chromadb.Client(chromadb.Settings(
        persist_directory=CHROMA_PERSIST_DIR,
        anonymized_telemetry=False
    ))

    # Delete existing collection if it exists, then recreate
    try:
        client.delete_collection(CHROMA_COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=CHROMA_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    # Embed and add documents
    texts = [doc["content"] for doc in documents]
    ids = [doc["id"] for doc in documents]
    embeddings = model.encode(texts).tolist()

    collection.add(
        documents=texts,
        embeddings=embeddings,
        ids=ids,
        metadatas=[{"filename": doc["filename"]} for doc in documents]
    )

    return collection


def query_collection(collection: chromadb.Collection, model: SentenceTransformer, query: str, top_k: int = 3) -> list[dict]:
    """Query ChromaDB for the top-k most similar chunks."""
    query_embedding = model.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    retrieved = []
    for i in range(len(results["ids"][0])):
        retrieved.append({
            "id": results["ids"][0][i],
            "content": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i]
        })

    return retrieved

