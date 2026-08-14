
# Module 3 — Zepto Support Assistant

A RAG-powered customer support assistant for Zepto, built with LangGraph, FastAPI, ChromaDB, and sentence-transformers.

---

## Architecture Description

### RAG Pipeline Stages

┌──────────────┐ ┌──────────────┐ ┌──────────────────┐ ┌──────────────────┐ │ INGESTION │───▶│ EMBEDDING │───▶│ RETRIEVAL │───▶│ GENERATION │ │ │ │ │ │ │ │ │ │ docs/*.txt │ │ SentenceTrans│ │ ChromaDB cosine │ │ Mock/LLM answer │ │ load_documents│ │ all-MiniLM │ │ query top-3 │ │ Pydantic output │ └──────────────┘ └──────────────┘ └──────────────────┘ └──────────────────┘


#### Stage 1: Ingestion
- **Component**: `app/embeddings.py` → `load_documents()`
- **What it does**: Reads all 8 `.txt` files from the `docs/` directory. Each file becomes one chunk (document-level chunking, appropriate given their short length).
- **Data flow**: Files on disk → list of `{id, content, filename}` dicts

#### Stage 2: Embedding
- **Component**: `app/embeddings.py` → `get_embedding_model()` + `initialize_chromadb()`
- **What it does**: Uses the `all-MiniLM-L6-v2` sentence-transformers model to generate 384-dimensional embeddings for each document. Stores embeddings, texts, and metadata in a ChromaDB collection named `zepto_policies` with cosine similarity.
- **Data flow**: Document texts → embedding vectors → ChromaDB collection

#### Stage 3: Retrieval
- **Component**: `app/embeddings.py` → `query_collection()` (called from `app/graph.py` → `retrieve_and_answer` node)
- **What it does**: Embeds the user's query with the same model, then queries ChromaDB for the top-3 most similar chunks via cosine similarity.
- **Data flow**: User query → query embedding → ChromaDB cosine search → top-3 chunks
- **MOCK_LLM toggle**: This stage does NOT branch — retrieval always runs for real in both modes.

#### Stage 4: Generation
- **Component**: `app/graph.py` → `retrieve_and_answer()` or `direct_answer()` nodes
- **What it does**: Produces the final answer based on retrieved context (policy questions) or a canned response (general questions).
- **MOCK_LLM toggle**: **This stage BRANCHES:**
  - **Default (MOCK_LLM=1)**: Returns a deterministic canned response — no LLM call. `retrieve_and_answer` returns `"Based on the retrieved context: {first 200 chars of top chunk}"`. `direct_answer` returns `"I can only answer questions about Zepto policies right now."`
  - **Optional (MOCK_LLM=0)**: Calls the real LLM (Groq) with the structured prompt template, validates output against the Pydantic schema, and retries up to 2 times on failure.

### LangGraph Flow

[START] → [classify_intent] ──┬── policy_question ──→ [retrieve_and_answer] → [END] │ └── general_question ─→ [direct_answer] ────→ [END]


- **classify_intent** (Node 1): Keyword heuristic in mock mode; LLM classification in real mode
- **retrieve_and_answer** (Node 2): Real retrieval + mock/real generation
- **direct_answer** (Node 3): Fixed canned string in mock mode; LLM in real mode
- **Conditional edge**: Routes based on `state["intent"]` — this routing logic does NOT depend on MOCK_LLM

### MOCK_LLM Summary

| Stage | MOCK_LLM=1 (default, graded) | MOCK_LLM=0 (optional) |
|-------|------------------------------|----------------------|
| classify_intent | Keyword heuristic | LLM call |
| Retrieval | Real (ChromaDB) | Real (ChromaDB) |
| Generation | Canned templates | LLM + Pydantic validation + retry |

---

## Example Calls (MOCK_LLM=1, graded baseline)

### Example 1: Policy question (triggers retrieval)

```bash
curl -X POST http://localhost:7860/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the delivery fee?"}'
Response:
{
  "answer": "Based on the retrieved context: Zepto delivers grocery and household essentials to serviceable pin codes within 10 to 30 minutes of order confirmation, depending on the customer's delivery zone and c",
  "sources": ["doc_01", "doc_05", "doc_04"],
  "confidence": 1.0
}

Explanation: The query contains "delivery" → classified as policy_question → routed to retrieve_and_answer → ChromaDB retrieval finds doc_01 (Delivery Policy) as the top match → mock answer built from first 200 characters of doc_01.

Example 2: General question (no retrieval)
curl -X POST http://localhost:7860/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the weather today?"}'
Response:
{
  "answer": "I can only answer questions about Zepto policies right now.",
  "sources": [],
  "confidence": 1.0
}

Explanation: The query contains none of the policy keywords → classified as general_question → routed to direct_answer → returns fixed canned string with empty sources.

Example 3: Policy question about refunds
curl -X POST http://localhost:7860/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "How do I get a refund for a damaged item?"}'
Response:

{
  "answer": "Based on the retrieved context: If an order arrives with damaged, spoiled, or missing items, customers must report it within 24 hours of delivery through the 'Report an Issue' button on the order pa",
  "sources": ["doc_06", "doc_02", "doc_05"],
  "confidence": 1.0
}

Explanation: The query contains "refund" → classified as policy_question → retrieval finds doc_06 (Damaged/Missing Items) as top match.

Running Locally
Prerequisites
Python 3.11+
pip
(Optional) Docker Desktop for containerized run
Setup & Run

cd support_assistant
python -m venv venv

# Activate virtual environment
# Windows PowerShell:
.\venv\Scripts\Activate.ps1
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the app (MOCK_LLM defaults to 1)
uvicorn app.main:app --host 0.0.0.0 --port 7860

The first run will download the all-MiniLM-L6-v2 model (~80MB). Subsequent runs use the cached model.
Test the endpoints

# Policy question
curl -X POST http://localhost:7860/ask -H "Content-Type: application/json" -d '{"query": "What is the delivery fee?"}'

# General question
curl -X POST http://localhost:7860/ask -H "Content-Type: application/json" -d '{"query": "What is the weather today?"}'

# Health check
curl http://localhost:7860/health


Docker
The Dockerfile builds and runs the app locally. No push to any registry is required.
Build & Run
cd support_assistant

# Build the image
docker build -t zepto-support-assistant .

# Run the container
docker run -p 7860:7860 zepto-support-assistant


Dockerfile Details
-Base image: python:3.11-slim
-Installs: All dependencies from requirements.txt
-Copies: docs/ (corpus) and app/ (application code)
-Environment: Sets MOCK_LLM=1 by default
-Serves: Port 7860 via uvicorn app.main:app --host 0.0.0.0 --port 7860

Verify:
curl -X POST http://localhost:7860/ask -H "Content-Type: application/json" -d '{"query": "How do I cancel my order?"}'

Project Structure
support_assistant/
├── docs/
│   ├── doc_01.txt          # Delivery Policy
│   ├── doc_02.txt          # Returns & Refunds
│   ├── doc_03.txt          # Membership Tiers
│   ├── doc_04.txt          # Order Tracking
│   ├── doc_05.txt          # Order Cancellation Policy
│   ├── doc_06.txt          # Damaged or Missing Items
│   ├── doc_07.txt          # Gift Cards
│   └── doc_08.txt          # Customer Support Hours
├── app/
│   ├── __init__.py         # Package init
│   ├── config.py           # Configuration & constants
│   ├── models.py           # Pydantic request/response models
│   ├── embeddings.py       # Document loading, embedding, ChromaDB
│   ├── prompt_template.py  # Structured prompt (role/context/task/format/length)
│   ├── graph.py            # LangGraph StateGraph (3 nodes + conditional edge)
│   └── main.py             # FastAPI app with /ask endpoint
├── Dockerfile              # Container config (locally buildable & runnable)
├── requirements.txt        # Python dependencies
└── README.md               # This file

Optional: Real LLM Extension (Ungraded)
To use Groq's free-tier LLM instead of mock mode:

export MOCK_LLM=0
export GROQ_API_KEY=your_groq_api_key_here
uvicorn app.main:app --host 0.0.0.0 --port 7860

-Provider: Groq (console.groq.com)
-Model: llama3-8b-8192
-Tier: Free (no credit card required)
-Note: Has request-rate quota limits