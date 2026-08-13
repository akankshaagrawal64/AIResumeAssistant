import os
import hashlib

from dotenv import load_dotenv
import streamlit as st

from graph.graph import graph

# ---------------------------------------------
# Load Environment
# ---------------------------------------------

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
# ---------------------------------------------
# Streamlit Config
# ---------------------------------------------

st.set_page_config(page_title="AI Resume Assistant")

st.title("AI Resume Assistant")
st.write("Upload your resume")

# ---------------------------------------------
# Upload Resume
# ---------------------------------------------

uploaded_file = st.file_uploader(
    "Choose a Resume",
    type=["pdf"]
)

if uploaded_file is not None:

    upload_folder = "uploads"
    os.makedirs(upload_folder, exist_ok=True)

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

    resume_hash = hashlib.md5(
        file_bytes
    ).hexdigest()

    st.success("Resume uploaded")

    config = {
        "configurable": {
            "thread_id": resume_hash
        }
    }

    # -----------------------------------------
    # Load existing history
    # -----------------------------------------

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
        "Ask another question",
        key=f"question_{len(history)}"
    )

    if question:

        with st.spinner("Thinking..."):

            result = graph.invoke(
                {
                    "file_path": file_path,
                    "resume_hash": resume_hash,
                    "question": question,
                    "documents": [],
                    "answer": "",
                    "chat_history": []
                },
                config=config
            )

        # Refresh page so the new Q&A appears above
        st.rerun()