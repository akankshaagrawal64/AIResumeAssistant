# AI Resume Assistant

An AI-powered Resume Assistant built with **Streamlit**, **LangChain**, **OpenAI**, and **ChromaDB**. Upload your resume in PDF format and ask questions about its content using a Retrieval-Augmented Generation (RAG) pipeline.

## Features

- Upload resumes in PDF format
- Extract and process resume text
- Split documents into manageable chunks
- Generate OpenAI embeddings
- Store embeddings in ChromaDB
- Retrieve relevant resume sections using semantic search
- Ask natural language questions about the resume
- Interactive Streamlit web interface

## Tech Stack

- Python
- Streamlit
- LangChain
- OpenAI API
- ChromaDB
- PyPDFLoader
- RecursiveCharacterTextSplitter

## Project Structure

```
aiResume/
│
├── app.py
├── .env
├── uploads/
├── chroma_db/
├── README.md
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/ai-resume-assistant.git
cd ai-resume-assistant
```
### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure OpenAI API Key

Create a `.env` file in the project root.

```
OPENAI_API_KEY=your_openai_api_key
```

## Running the Application

Start the Streamlit server.

```bash
streamlit run app.py
```
Or 
python -m streamlit run app.py

The application will open in your browser.

## How It Works

1. Upload a resume in PDF format.
2. The PDF is loaded using **PyPDFLoader**.
3. The document is split into chunks using **RecursiveCharacterTextSplitter**.
4. OpenAI generates embeddings for each chunk.
5. Embeddings are stored in **ChromaDB**.
6. User questions are converted into embeddings.
7. Relevant chunks are retrieved using similarity search.
8. The retrieved context and user question are sent to the LLM.
9. The AI generates an accurate response based on the resume.

## Example Questions

- Tell me about this candidate.
- What programming languages does the candidate know?
- Summarize the work experience.
- What projects has the candidate completed?
- List the candidate's technical skills.
- Does the candidate have experience with LangChain?
- What certifications are mentioned?


## Requirements

- Python 3.10+
- OpenAI API Key

Built using Streamlit, LangChain, OpenAI and ChromaDB.
