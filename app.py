import os
import uuid
import hashlib
import streamlit as st

from graph.graph import graph
from graph.ingestion import ingestion

# ---------------------------------------------
# Streamlit Config
# ---------------------------------------------

st.set_page_config(page_title="AI Document Assistant")

st.title("AI Document Assistant")
st.write("Upload your Document")


# ---------------------------------------------
# Create Session ID
# ---------------------------------------------

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())


session_id = st.session_state.session_id
namespace = f"session-{session_id}"
# ---------------------------------------------
# Upload Document
# ---------------------------------------------

uploaded_files = st.file_uploader(
    "Choose a Document",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files is not None:
    upload_folder = "uploads"
    os.makedirs(upload_folder, exist_ok=True)
    for uploaded_file in uploaded_files:
        file_path = os.path.join(
            upload_folder,
            uploaded_file.name
        )
        # Save PDF
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # -----------------------------------------
        # Create Resume Hash
        # -----------------------------------------

        file_bytes = uploaded_file.getvalue()

        document_id = hashlib.md5(
            file_bytes
        ).hexdigest()

         # -----------------------------------------
        # Process document
        # -----------------------------------------

        with st.spinner(
            f"Processing {uploaded_file.name}..."
        ):

            ingestion.invoke({

                "file_path": file_path,

                "document_id": document_id,

                "document_name": uploaded_file.name,

                "vectorstore_namespace": namespace,

                "question": "",

                "documents": [],

                "reranked_documents": [],

                "answer": "",

                "chat_history": []
            })


        st.success(
            f"{uploaded_file.name} processed"
        )


    # -----------------------------------------
    # Load existing history
    # -----------------------------------------
    
    config = {
        "configurable": {
            "thread_id": session_id
        }
    }
    state = graph.get_state(config)

    history = []

    if state.values:
        history = state.values.get("chat_history", [])

    # -----------------------------------------
    # Show previous conversation
    # -----------------------------------------

    for i, chat in enumerate(history, 1):
        st.markdown(f"### Question {i}")
        st.write(chat["question"])

        st.markdown("**Answer**")
        st.write(chat["answer"])

        st.divider()

    # -----------------------------------------
    # Ask next question
    # -----------------------------------------

    question = st.text_input(
        "Ask question",
        key=f"question_{len(history)}"
    )

    if question:

        with st.spinner("Thinking..."):

            result = graph.invoke(

                {
                    "vectorstore_namespace": namespace,

                    "question": question,

                    "documents": [],

                    "reranked_documents": [],

                    "answer": "",

                    "chat_history": []
                },

                config=config
            )

        
        # Refresh page so the new Q&A appears above
        st.rerun()