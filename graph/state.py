from typing import TypedDict, Annotated
from operator import add


class ResumeState(TypedDict):

    # Current document
    file_path: str
    document_id: str
    document_name: str

    # Pinecone
    vectorstore_namespace: str

    # Question / answer
    question: str
    documents: list
    reranked_documents: list
    answer: str

    # Conversation
    chat_history: Annotated[list, add]