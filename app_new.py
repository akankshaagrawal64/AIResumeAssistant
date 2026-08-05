import os
import hashlib

from dotenv import load_dotenv
import streamlit as st

from graph.graph import graph


# ---------------------------------------------
# Load Environment
# ---------------------------------------------

load_dotenv()


# ---------------------------------------------
# Streamlit Config
# ---------------------------------------------

st.set_page_config(
    page_title="AI Resume Assistant"
)

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

    os.makedirs(
        upload_folder,
        exist_ok=True
    )


    file_path = os.path.join(
        upload_folder,
        uploaded_file.name
    )


    # Save PDF

    with open(file_path, "wb") as f:
        f.write(
            uploaded_file.getbuffer()
        )


    # -----------------------------------------
    # Create Resume Hash
    # -----------------------------------------

    file_bytes = uploaded_file.getvalue()

    resume_hash = hashlib.md5(
        file_bytes
    ).hexdigest()


    st.success(
        "Resume uploaded"
    )


    # -----------------------------------------
    # Ask Question
    # -----------------------------------------

    st.divider()


    question = st.text_input(
        "Ask a question about your resume"
    )


    if question:


        config = {
            "configurable": {
                "thread_id": resume_hash
            }
        }


        with st.spinner(
            "Thinking..."
        ):


            result = graph.invoke(
                {
                    "file_path": file_path,
                    "resume_hash": resume_hash,
                    "question": question,
                    "documents": [],
                    "answer": ""
                },
                config=config
            )


        st.subheader(
            "Answer"
        )

        st.write(
            result["answer"]
        )