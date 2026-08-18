from langgraph.graph import StateGraph, START, END

from graph.state import ResumeState
from graph.nodes import process_resume


builder = StateGraph(ResumeState)

builder.add_node(
    "process_documents",
    process_resume
)

builder.add_edge(
    START,
    "process_documents"
)

builder.add_edge(
    "process_documents",
    END
)

ingestion = builder.compile()