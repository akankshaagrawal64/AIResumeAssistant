from langgraph.graph import StateGraph, START, END

from graph.state import ResumeState
from langgraph.checkpoint.memory import InMemorySaver



from graph.nodes import (
    process_resume,
    retrieve,
    answer
)


builder = StateGraph(ResumeState)


# Nodes

builder.add_node(
    "process_resume",
    process_resume
)

builder.add_node(
    "retrieve",
    retrieve
)

builder.add_node(
    "answer",
    answer
)


# Flow

builder.add_edge(
    START,
    "process_resume"
)

builder.add_edge(
    "process_resume",
    "retrieve"
)

builder.add_edge(
    "retrieve",
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