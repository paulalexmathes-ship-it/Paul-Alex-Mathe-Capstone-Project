
"""FastAPI application wrapping the LangGraph RAG pipeline."""
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager

from app.config import MOCK_LLM
from app.models import QueryRequest, QueryResponse
from app.embeddings import load_documents, get_embedding_model, initialize_chromadb
from app.graph import build_graph

# Global references
_collection = None
_model = None
_graph = None


def get_collection():
    """Get the ChromaDB collection."""
    return _collection


def get_model():
    """Get the embedding model."""
    return _model


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize resources on startup."""
    global _collection, _model, _graph

    print("🚀 Initializing Support Assistant...")
    print(f"   MOCK_LLM mode: {'ON (graded baseline)' if MOCK_LLM else 'OFF (real LLM)'}")

    # Load embedding model
    print("   Loading embedding model (all-MiniLM-L6-v2)...")
    _model = get_embedding_model()

    # Load and embed documents
    print("   Loading and embedding documents...")
    documents = load_documents()
    _collection = initialize_chromadb(documents, _model)
    print(f"   ✅ {len(documents)} documents embedded and indexed in ChromaDB")

    # Build LangGraph
    print("   Building LangGraph pipeline...")
    _graph = build_graph()
    print("   ✅ LangGraph compiled successfully")

    print("🟢 Support Assistant ready!")
    yield

    # Cleanup
    print("🔴 Shutting down Support Assistant...")


app = FastAPI(
    title="Zepto Support Assistant",
    description="RAG-powered support assistant for Zepto policies",
    version="1.0.0",
    lifespan=lifespan
)


@app.post("/ask", response_model=QueryResponse)
async def ask(request: QueryRequest):
    """
    POST /ask endpoint.
    Accepts a query and returns a structured JSON response with answer, sources, and confidence.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    # Run the LangGraph pipeline
    initial_state = {
        "query": request.query,
        "intent": "",
        "retrieved_chunks": [],
        "response": {}
    }

    result = _graph.invoke(initial_state)

    # Validate and return response
    response_data = result["response"]
    return QueryResponse(
        answer=response_data["answer"],
        sources=response_data["sources"],
        confidence=response_data["confidence"]
    )


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "mock_llm": MOCK_LLM}

