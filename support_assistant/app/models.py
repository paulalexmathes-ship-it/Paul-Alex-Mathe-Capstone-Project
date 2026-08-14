
"""Pydantic models for request/response schema."""
from pydantic import BaseModel, Field
from typing import List


class QueryRequest(BaseModel):
    """Request model for the /ask endpoint."""
    query: str = Field(..., description="The user's question")


class QueryResponse(BaseModel):
    """Structured response model with answer, sources, and confidence."""
    answer: str = Field(..., description="The generated answer")
    sources: List[str] = Field(default_factory=list, description="List of chunk/document IDs used")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score between 0 and 1")

