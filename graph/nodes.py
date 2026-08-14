import streamlit as st
import os
import hashlib
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
#from langchain_openai.embeddings import OpenAIEmbeddings
#from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pc = Pinecone(
    api_key=pinecone_api_key
)

index_name = "doc-assistant"

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

index = pc.Index(index_name)

#embedding = OpenAIEmbeddings(model = "text-embedding-ada-002", api_key=api_key)
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

chat = ChatOpenAI(
    model="gpt-4",
    temperature=0,
    api_key=api_key
)

template = """
Answer the question using ONLY the resume context.

Question:
{question}

Context:
{context}
"""

prompt = PromptTemplate.from_template(template)

def process_resume(state):

    file_path = state["file_path"]

    # Create hash
    with open(file_path, "rb") as f:
        file_bytes = f.read()

    resume_hash = hashlib.md5(
        file_bytes
    ).hexdigest()


    namespace = f"doc-{resume_hash}"

    # -----------------------------
    # Already processed?
    # -----------------------------
    '''
    if (
        state.get("resume_hash") == resume_hash
        and state.get("vectorstore_namespace") == namespace
    ):
        return {
            "resume_hash": resume_hash,
            "vectorstore_namespace": namespace
        }
    '''
    # ---------------------------------
    # Check Pinecone
    # ---------------------------------

    stats = index.describe_index_stats()

    namespace_info = stats.namespaces.get(namespace)

    if namespace_info and namespace_info.vector_count > 0:

        # Already embedded
        return {
            "resume_hash": resume_hash,
            "vectorstore_namespace": namespace
        }


    # Load PDF

    loader = PyPDFLoader(file_path)

    pages = loader.load()


    # Split

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )


    chunks = splitter.split_documents(
        pages
    )


    # Create Vector DB

    #vectorstore_path = "./chroma_db"

    '''
    Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory=vectorstore_path
    )

    '''

    # -----------------------------
    # Add documents to Pinecone
    # -----------------------------

    PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embedding,
        index_name="doc-assistant",
        namespace=namespace
    )
     # Give Pinecone a little time
    time.sleep(2)
    return {
        "resume_hash": resume_hash,
        #"vectorstore_path": vectorstore_path
        "vectorstore_namespace": namespace
    }



def retrieve(state):
    '''
    vectorstore = Chroma(
        persist_directory=state["vectorstore_path"],
        embedding_function=embedding
    )
    '''

    vectorstore = PineconeVectorStore(
        index_name="doc-assistant",
        embedding=embedding,
        namespace=state["vectorstore_namespace"]
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k":3
        }
    )


    docs = retriever.invoke(
        state["question"]
    )


    return {
        "documents": docs
    }


def answer(state):
    context = "\n\n".join(
        doc.page_content for doc in state["documents"]
    )

    response = chat.invoke(
        prompt.format(
            question=state["question"],
            context=context
        )
    )
   
    return {
        "answer": response.content,
        "chat_history": [
            {
                "question": state["question"],
                "answer": response.content,
            }
        ]
    }