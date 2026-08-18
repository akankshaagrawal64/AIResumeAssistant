# AI Document Assistant

An AI-powered Document Assistant built with **Streamlit, LangGraph, LangChain, Pinecone, Sentence Transformers, CrossEncoder, OpenAI, and RAGAS**.

The application allows users to upload multiple PDF documents and ask natural-language questions about their content using a **Retrieval-Augmented Generation (RAG)** pipeline.

The system uses:

* **Pinecone** as the vector database
* **Sentence Transformers** for document embeddings
* **CrossEncoder** for document reranking
* **OpenAI GPT** for answer generation
* **LangGraph** for workflow and state management
* **RAGAS** for RAG evaluation

---

## Features

* Upload multiple PDF documents
* Extract text from PDF documents
* Split documents into smaller chunks
* Generate embeddings using Sentence Transformers
* Store document embeddings in Pinecone
* Use Pinecone namespaces for session-level document isolation
* Store document and page metadata with each vector
* Retrieve relevant chunks using semantic similarity
* Rerank retrieved chunks using CrossEncoder
* Generate answers using OpenAI GPT
* Support multiple documents in a single session
* Avoid unnecessary re-embedding of already processed documents
* Document-level and page-level source metadata
* Source/citation support
* RAG evaluation using RAGAS
* Evaluate retrieval quality and answer quality

---

# Architecture

The application consists of two major workflows:

1. **Document Processing**
2. **Question Answering**

### High-Level Architecture

```text
                         USER
                          |
                          v
                   Streamlit UI
                          |
             +------------+------------+
             |                         |
             v                         v
       Upload Documents           Ask Question
             |                         |
             v                         v
       Generate Session ID       LangGraph Workflow
             |                         |
             v                         v
      Process Documents            Retrieve
             |                         |
             v                         v
          Pinecone              Pinecone Search
                                       |
                                       v
                                  Top 7 Chunks
                                       |
                                       v
                                  CrossEncoder
                                       |
                                       v
                                   Top 3 Chunks
                                       |
                                       v
                                   OpenAI GPT
                                       |
                                       v
                                    Answer
```

---

# RAG Pipeline

The question-answering pipeline works as follows:

```text
User Question
      |
      v
Sentence Transformer
      |
      v
Question Embedding
      |
      v
Pinecone Similarity Search
      |
      v
Retrieve Top 7 Chunks
      |
      v
CrossEncoder Reranking
      |
      v
Select Top 3 Chunks
      |
      v
Build Context
      |
      v
OpenAI GPT
      |
      v
Final Answer
```

The purpose of the pipeline is to retrieve the most relevant information from the uploaded documents before sending the context to the LLM.

---

# Document Processing Pipeline

When a PDF document is uploaded, it goes through the following pipeline:

```text
PDF
 |
 v
PyPDFLoader
 |
 v
Text Extraction
 |
 v
RecursiveCharacterTextSplitter
 |
 v
Document Chunks
 |
 v
Metadata Added
 |
 v
Sentence Transformer
 |
 v
Vector Embeddings
 |
 v
Pinecone
```

---

# Embeddings

The project uses the following Sentence Transformers model:

```text
sentence-transformers/all-MiniLM-L6-v2
```

It is used through LangChain's `HuggingFaceEmbeddings`:

```python
HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
```

The embedding dimension is:

```text
384
```

Therefore, the Pinecone index is configured with:

```text
Dimension: 384
Metric: cosine
```

---

# Pinecone

[Pinecone](https://www.pinecone.io/) is used as the vector database.

### Index Configuration

```text
Index Name: doc-assistant
Dimension: 384
Metric: cosine
Cloud: AWS
Region: us-east-1
```

The application stores document vectors in Pinecone instead of using a local vector database such as ChromaDB.

---

# Pinecone Namespace

Each application session receives a unique session ID.

Example:

```python
session_id = str(uuid.uuid4())
```

The namespace is generated as:

```python
namespace = f"session-{session_id}"
```

Example:

```text
session-550e8400-e29b-41d4-a716-446655440000
```

This allows multiple documents belonging to the same session to be searched together while keeping documents from different sessions isolated.

---

# Multiple Document Support

The application supports uploading multiple PDF documents:

```python
uploaded_files = st.file_uploader(
    "Choose Documents",
    type=["pdf"],
    accept_multiple_files=True
)
```

Each uploaded document receives its own:

* Document ID
* Document name
* Document hash
* Metadata
* Chunks

All documents uploaded within the same session can be searched together.

---

# Metadata

Metadata is stored with each document chunk.

Example:

```python
chunk.metadata.update({
    "document_id": document_id,
    "document_name": document_name,
    "page": page_number,
    "chunk_id": chunk_id
})
```

Example metadata:

```text
document_id   : abc123
document_name : resume.pdf
page          : 4
chunk_id      : 17
```

Metadata allows the application to identify where retrieved content originated.

---


# Reranking

The system initially retrieves several potentially relevant chunks from Pinecone.

### Retrieval Configuration

```text
Initial Retrieval: Top 7
```

The retrieved chunks are then passed to a CrossEncoder.

### CrossEncoder Model

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

The CrossEncoder evaluates each:

```text
(question, document)
```

pair.

Example:

```text
Question
   |
   +-- Document 1 -> 0.91
   +-- Document 2 -> 0.73
   +-- Document 3 -> 0.95
   +-- Document 4 -> 0.41
   +-- Document 5 -> 0.82
```

The documents are sorted according to their relevance scores.

The application then selects the top 3 chunks.

```text
Pinecone
   |
   v
Top 7
   |
   v
CrossEncoder
   |
   v
Top 3
   |
   v
LLM
```

Reranking improves the quality of the context provided to the LLM by filtering out less relevant retrieved chunks.

---

# LangGraph

LangGraph is used to manage the application workflow and state.

The main question-answering workflow is:

```text
START
  |
  v
retrieve
  |
  v
rerank
  |
  v
answer
  |
  v
END
```

The main nodes are:

```text
process_resume
retrieve
rerank
answer
```

---

# Evaluation

The project uses **RAGAS** to evaluate the RAG pipeline.

The evaluation focuses on four important areas:

1. Context Recall
2. Context Precision
3. Faithfulness
4. Answer Relevancy

---

# Evaluation Pipeline

The evaluation architecture is:

```text
Test Questions
      |
      v
RAG Pipeline
      |
      v
Pinecone Retrieval
      |
      v
CrossEncoder Reranking
      |
      v
OpenAI Answer
      |
      v
RAGAS Evaluation
      |
      +-----------------------+
      |                       |
      v                       v
Retrieval Metrics        Answer Metrics
      |                       |
      +-----------+-----------+
                  |
                  v
         Evaluation Results
```

---

# RAGAS Metrics

| Metric            | What it measures                                     |
| ----------------- | ---------------------------------------------------- |
| Context Recall    | Whether the required information was retrieved       |
| Context Precision | Whether retrieved information is relevant            |
| Faithfulness      | Whether the answer is supported by retrieved context |
| Answer Relevancy  | Whether the answer addresses the question            |

---

# Project Structure

```text
aiResume/
│
├── app.py
│
├── graph/
│   ├── __init__.py
│   ├── graph.py
│   ├── nodes.py
│   └── state.py
│
├── evaluation/
│   ├── __init__.py
│   ├── evaluate.py
│   └── test_dataset.py
│
├── uploads/
│
├── .env
├── .gitignore
├── requirements.txt
└── README.md
```

> `uploads/` should generally be excluded from Git because uploaded documents may contain private information.

---

# Environment Variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
```

---

# Installation

## 1. Clone the Repository

```bash
git clone https://github.com/your-username/ai-resume-assistant.git
```

Go into the project directory:

```bash
cd ai-resume-assistant
```

---

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv .venv
```

Activate the virtual environment:

```bash
.venv\Scripts\activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Running the Application

Start Streamlit:

```bash
streamlit run app.py
```

If the `streamlit.exe` command is blocked by Windows Application Control Policy, use:

```bash
python -m streamlit run app.py
```

The application will open in your browser.

---

# Running Evaluation

The evaluation code is located in:

```text
evaluation/evaluate.py
```

Run it from the project root:

```bash
python -m evaluation.evaluate
```

The evaluation dataset is maintained in:

```text
evaluation/test_dataset.py
```

---

# Technologies Used

| Technology                     | Purpose                         |
| ------------------------------ | ------------------------------- |
| Python                         | Application development         |
| Streamlit                      | User interface                  |
| LangGraph                      | Workflow and state management   |
| LangChain                      | LLM and RAG components          |
| OpenAI                         | Answer generation               |
| Sentence Transformers          | Text embeddings                 |
| Pinecone                       | Vector database                 |
| CrossEncoder                   | Document reranking              |
| PyPDFLoader                    | PDF extraction                  |
| RecursiveCharacterTextSplitter | Document chunking               |
| RAGAS                          | RAG evaluation                  |
| python-dotenv                  | Environment variable management |

---

# Models

## Embedding Model

```text
sentence-transformers/all-MiniLM-L6-v2
```

Embedding dimension:

```text
384
```

---

## Reranking Model

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

---

## LLM

```text
GPT-4
```

---

# Data Flow

## Document Ingestion

```text
PDF
 ↓
PyPDFLoader
 ↓
Text Extraction
 ↓
RecursiveCharacterTextSplitter
 ↓
Document Chunks
 ↓
Metadata
 ↓
Sentence Transformer
 ↓
384-Dimensional Vector
 ↓
Pinecone
```

---

## Question Answering

```text
Question
 ↓
Sentence Transformer
 ↓
Question Embedding
 ↓
Pinecone
 ↓
Top 7 Chunks
 ↓
CrossEncoder
 ↓
Top 3 Chunks
 ↓
Context Construction
 ↓
GPT
 ↓
Answer + Sources
```

---

## Evaluation

```text
Question
 ↓
RAG Pipeline
 ↓
Retrieved Context
 ↓
Generated Answer
 ↓
RAGAS
 ↓
Context Recall
Context Precision
Faithfulness
Answer Relevancy
```

---

# Requirements

The project requires:

* Python 3.10+
* OpenAI API key
* Pinecone API key
* Internet connection for API/model access
* Pinecone account

Python dependencies are listed in:

```text
requirements.txt
```

Install them using:

```bash
pip install -r requirements.txt
```

---


# License

This project is intended for **learning and development purposes**.

---

# Built With

```text
Python
Streamlit
LangGraph
LangChain
Pinecone
Sentence Transformers
CrossEncoder
OpenAI
RAGAS
```
