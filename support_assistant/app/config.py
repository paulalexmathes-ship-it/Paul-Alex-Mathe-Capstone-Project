
"""Configuration module for the Support Assistant."""
import os

# MOCK_LLM toggle: unset or "1" = mock mode (graded baseline), "0" = real LLM
MOCK_LLM = os.environ.get("MOCK_LLM", "1") != "0"

# ChromaDB settings
CHROMA_COLLECTION_NAME = "zepto_policies"
CHROMA_PERSIST_DIR = "./chroma_db"

# Embedding model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Document directory
DOCS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs")

# Keywords for intent classification (mock mode)
POLICY_KEYWORDS = [
    "delivery", "return", "refund", "membership",
    "tracking", "cancel", "gift card", "support hours"
]

# Optional: Groq API settings (only used when MOCK_LLM=0)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = "llama3-8b-8192"

