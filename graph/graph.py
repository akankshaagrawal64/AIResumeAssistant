from langgraph.graph import StateGraph, START, END

from langgraph.checkpoint.memory import InMemorySaver

from graph.state import ResumeState

from graph.nodes import (
    retrieve,
    rerank,
    answer
)


builder = StateGraph(ResumeState)


builder.add_node(
    "retrieve",
    retrieve
)

builder.add_node(
    "rerank",
    rerank
)

builder.add_node(
    "answer",
    answer
)


builder.add_edge(
    START,
    "retrieve"
)

builder.add_edge(
    "retrieve",
    "rerank"
)

builder.add_edge(
    "rerank",
    "answer"
)

builder.add_edge(
    "answer",
    END
)


memory = InMemorySaver()

graph = builder.compile(
    checkpointer=memory
)