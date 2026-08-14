from typing import TypedDict, Annotated
from operator import add
class ResumeState(TypedDict):
    file_path: str
    resume_hash: str
    vectorstore_namespace: str
    question: str
    documents: list
    answer: str
    chat_history: Annotated[list, add]
