#kiarash

from typing import Literal, TypedDict
from uuid import UUID
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from app.rag import RAGService
from app.schemas import Source

class ChatState(TypedDict):
    document_id: UUID
    question: str
    sources: list[Source]
    answer: str
    status: Literal["answered", "no_context"]


def build_graph(service: RAGService) -> CompiledStateGraph:
    def retrieve(state: ChatState) -> dict:
        return {"sources": service.retrieve(state["document_id"], state["question"])}

    def route(state: ChatState) -> Literal["generate", "fallback"]:
        return "generate" if state["sources"] else "fallback"

    def generate(state: ChatState) -> dict:
        return {
            "answer": service.provider.answer(state["question"], state["sources"]),
            "status": "answered",
        }

    def fallback(state: ChatState) -> dict:
        return {
            "answer": "I could not find relevant evidence in this document.",
            "status": "no_context",
        }

    graph = StateGraph(ChatState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.add_node("fallback", fallback)
    graph.add_edge(START, "retrieve")
    graph.add_conditional_edges("retrieve", route)
    graph.add_edge("generate", END)
    graph.add_edge("fallback", END)
    return graph.compile()
