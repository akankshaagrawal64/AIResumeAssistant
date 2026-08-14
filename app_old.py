import os
from dotenv import load_dotenv
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnableParallel
from langchain_core.output_parsers import StrOutputParser

import copy
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
embedding = OpenAIEmbeddings(model = "text-embedding-ada-002", api_key=api_key)


st.set_page_config(page_title="AI Resume Assistant")
st.title("AI Resume Assistant")
st.write("Upload your resume")
uploaded_file = st.file_uploader(
    "Choose a resume",
    type =["pdf"]
)
if uploaded_file is not None:
    st.success("Resume uploaded successfully")
    upload_folder = "uploads"
    os.makedirs(upload_folder, exist_ok=True)

    # Save the uploaded file
    file_path = os.path.join(upload_folder, uploaded_file.name)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    loader_pdf = PyPDFLoader(file_path)
    pages_pdf = loader_pdf.load()
    pages_pdf_cut = copy.deepcopy(pages_pdf)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    for i in pages_pdf_cut:
        i.page_content = ' '.join(i.page_content.split())

    chunks = splitter.split_documents(pages_pdf_cut)

   # vectorstore = Chroma.from_documents( documents=chunks, embedding=embedding, persist_directory="./chroma_db")
    persist_directory = "./chroma_db"

    vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embedding,
            persist_directory=persist_directory,
        )
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )
    Template = '''
    Answer the following question:
    {question}

    To answer the question, use the following context:
    {context}
    
    '''
    prompt_template = PromptTemplate.from_template(Template)
    chat = ChatOpenAI(model_name = 'gpt-4', model_kwargs ={'seed' :365}, max_tokens =250,  api_key=api_key)
   
    st.divider()

    question = st.text_input("Ask a question about your resume")
    chain = {'context' :retriever, 'question' :RunnablePassthrough()} | prompt_template | chat | StrOutputParser()
    response = chain.invoke(question)
    st.write("### Answer")
    st.write(response)
    



