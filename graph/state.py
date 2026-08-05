from typing import TypedDict

class ResumeState(TypedDict):
    resume_hash: str
    vectorstore_path: str
    question: str
    documents: list
    answer: str