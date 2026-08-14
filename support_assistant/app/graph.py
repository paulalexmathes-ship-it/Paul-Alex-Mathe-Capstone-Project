
"""LangGraph StateGraph with 3 nodes and conditional routing."""
import json
from typing import TypedDict, List
from langgraph.graph import StateGraph, END

from app.config import MOCK_LLM, POLICY_KEYWORDS, GROQ_API_KEY, GROQ_MODEL
from app.models import QueryResponse
from app.embeddings import query_collection
from app.prompt_template import SYSTEM_PROMPT


class GraphState(TypedDict):
    """Typed state for the LangGraph StateGraph."""
    query: str
    intent: str
    retrieved_chunks: List[dict]
    response: dict


def classify_intent(state: GraphState) -> GraphState:
    """
    Node 1: Classify the query as policy_question or general_question.
    Mock mode: keyword heuristic. Real LLM mode: call LLM.
    """
    query_lower = state["query"].lower()

    if MOCK_LLM:
        # Graded baseline: keyword-based classification
        is_policy = any(keyword in query_lower for keyword in POLICY_KEYWORDS)
        state["intent"] = "policy_question" if is_policy else "general_question"
    else:
        # Optional extension: use real LLM for classification
        try:
            from groq import Groq
            client = Groq(api_key=GROQ_API_KEY)
            classification_prompt = (
                f"Classify the following customer query as either 'policy_question' "
                f"(if it's about delivery, returns, refunds, membership, tracking, "
                f"cancellation, gift cards, or support hours) or 'general_question' "
                f"(if it's unrelated to these topics). "
                f"Respond with ONLY 'policy_question' or 'general_question'.\n\n"
                f"Query: {state['query']}"
            )
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": classification_prompt}],
                temperature=0,
                max_tokens=20
            )
            result = response.choices[0].message.content.strip().lower()
            state["intent"] = "policy_question" if "policy" in result else "general_question"
        except Exception:
            # Fallback to keyword heuristic
            is_policy = any(keyword in query_lower for keyword in POLICY_KEYWORDS)
            state["intent"] = "policy_question" if is_policy else "general_question"

    return state


def retrieve_and_answer(state: GraphState) -> GraphState:
    """
    Node 2: For policy_question — retrieve top-3 chunks and generate answer.
    Retrieval always runs for real. Only generation branches on MOCK_LLM.
    """
    from app.main import get_collection, get_model

    collection = get_collection()
    model = get_model()

    # Retrieval always runs (no API key needed)
    retrieved = query_collection(collection, model, state["query"], top_k=3)
    state["retrieved_chunks"] = retrieved

    source_ids = [chunk["id"] for chunk in retrieved]

    if MOCK_LLM:
        # Graded baseline: canned templated answer
        top_chunk_snippet = retrieved[0]["content"][:200] if retrieved else ""
        answer = f"Based on the retrieved context: {top_chunk_snippet}"
        state["response"] = {
            "answer": answer,
            "sources": source_ids,
            "confidence": 1.0
        }
    else:
        # Optional extension: call real LLM with structured prompt
        context = "\n\n".join([
            f"[{chunk['id']}]: {chunk['content']}" for chunk in retrieved
        ])
        prompt = SYSTEM_PROMPT.format(context=context, query=state["query"])

        state["response"] = _call_llm_with_retry(prompt, source_ids)

    return state


def direct_answer(state: GraphState) -> GraphState:
    """
    Node 3: For general_question — answer without retrieval.
    Mock mode: fixed canned string. Real LLM mode: call LLM directly.
    """
    state["retrieved_chunks"] = []

    if MOCK_LLM:
        # Graded baseline: fixed canned response
        state["response"] = {
            "answer": "I can only answer questions about Zepto policies right now.",
            "sources": [],
            "confidence": 1.0
        }
    else:
        # Optional extension: call real LLM without retrieval
        try:
            from groq import Groq
            client = Groq(api_key=GROQ_API_KEY)
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": state["query"]}],
                temperature=0.7,
                max_tokens=200
            )
            answer_text = response.choices[0].message.content.strip()
            state["response"] = {
                "answer": answer_text,
                "sources": [],
                "confidence": 0.5
            }
        except Exception:
            state["response"] = {
                "answer": "I can only answer questions about Zepto policies right now.",
                "sources": [],
                "confidence": 1.0
            }

    return state


def _call_llm_with_retry(prompt: str, source_ids: List[str], max_retries: int = 2) -> dict:
    """
    Call LLM and validate against Pydantic schema.
    Retry up to max_retries times with corrective instruction on failure.
    """
    try:
        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY)
    except Exception:
        return {
            "answer": "Error: LLM service unavailable.",
            "sources": source_ids,
            "confidence": 0.0
        }

    messages = [{"role": "user", "content": prompt}]

    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                temperature=0,
                max_tokens=500
            )
            raw_output = response.choices[0].message.content.strip()

            # Try to parse as JSON and validate with Pydantic
            parsed = json.loads(raw_output)
            validated = QueryResponse(**parsed)
            return validated.model_dump()

        except (json.JSONDecodeError, Exception) as e:
            if attempt < max_retries:
                # Add corrective instruction
                corrective = (
                    f"Your previous response was not valid JSON. "
                    f"Please respond with ONLY a valid JSON object with fields: "
                    f'"answer" (string), "sources" (list of strings), "confidence" (float 0-1). '
                    f"Error: {str(e)}"
                )
                messages.append({"role": "assistant", "content": raw_output if 'raw_output' in dir() else ""})
                messages.append({"role": "user", "content": corrective})
            else:
                # Give up after max retries
                return {
                    "answer": "Error: Failed to generate valid response after retries.",
                    "sources": source_ids,
                    "confidence": 0.0
                }

    return {
        "answer": "Error: Unexpected failure.",
        "sources": source_ids,
        "confidence": 0.0
    }


def route_by_intent(state: GraphState) -> str:
    """Conditional edge: route based on classified intent."""
    if state["intent"] == "policy_question":
        return "retrieve_and_answer"
    else:
        return "direct_answer"


def build_graph() -> StateGraph:
    """Build and compile the LangGraph StateGraph."""
    workflow = StateGraph(GraphState)

    # Add nodes
    workflow.add_node("classify_intent", classify_intent)
    workflow.add_node("retrieve_and_answer", retrieve_and_answer)
    workflow.add_node("direct_answer", direct_answer)

    # Set entry point
    workflow.set_entry_point("classify_intent")

    # Add conditional edge from classify_intent
    workflow.add_conditional_edges(
        "classify_intent",
        route_by_intent,
        {
            "retrieve_and_answer": "retrieve_and_answer",
            "direct_answer": "direct_answer"
        }
    )

    # Both answer nodes go to END
    workflow.add_edge("retrieve_and_answer", END)
    workflow.add_edge("direct_answer", END)

    return workflow.compile()

