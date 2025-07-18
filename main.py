import streamlit as st
import openai
import os
from datetime import datetime

# Set your OpenAI API key
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Load Assistant ID from Streamlit secrets
ASSISTANT_ID = st.secrets["ASSISTANT_ID"]
VECTORSTORE_ID = st.secrets["VECTORSTORE_ID"]  # Add this to your Streamlit Secrets

st.set_page_config(page_title="Regulatory Assistant UI", layout="wide")
st.title("📄 AI Regulatory Document Assistant")

# Initialize session state
if "thread_id" not in st.session_state:
    thread = openai.beta.threads.create()
    st.session_state.thread_id = thread.id

# Sidebar file upload for vector store (persistent files)
st.sidebar.header("📚 Add to Knowledge Base")
persist_files = st.sidebar.file_uploader("Upload PDFs for vector store", type=["pdf"], accept_multiple_files=True)

if persist_files:
    for file in persist_files:
        try:
            uploaded = openai.files.create(file=file, purpose="assistants")
            openai.beta.vector_stores.file_batches.create(
                vector_store_id=VECTORSTORE_ID,
                file_ids=[uploaded.id]
            )
            st.sidebar.success(f"Uploaded {file.name} to vector store.")
        except Exception as e:
            st.sidebar.error(f"Error uploading {file.name}: {e}")

# Sidebar list of uploaded files
st.sidebar.markdown("---")
st.sidebar.subheader("📄 Files in Vector Store")
try:
    vector_files = openai.beta.vector_stores.files.list(vector_store_id=VECTORSTORE_ID).data
    if vector_files:
        for f in vector_files:
            upload_time = datetime.fromtimestamp(f.created_at).strftime("%Y-%m-%d %H:%M")
            download_url = f"https://api.openai.com/v1/files/{f.id}/content"
            col1, col2 = st.sidebar.columns([4, 1])
            with col1:
                st.markdown(f"[{f.filename} — {upload_time}]({download_url})")
            with col2:
                if st.button("🗑️", key=f"delete_{f.id}"):
                    if st.sidebar.checkbox(f"Confirm delete: {f.filename}", key=f"confirm_{f.id}"):
                        try:
                            openai.files.delete(f.id)
                            st.experimental_rerun()
                        except Exception as e:
                            st.sidebar.error(f"Error deleting {f.filename}: {e}")
    else:
        st.sidebar.info("No files in vector store yet.")
except Exception as e:
    st.sidebar.error("Error retrieving file list.")

# Temporary file uploads for one-off conversations
st.subheader("💬 Chat with your Assistant")
temp_files = st.file_uploader("Attach temporary files (for this conversation only)", type=["pdf"], accept_multiple_files=True)

# Text input for chat
user_input = st.chat_input("Ask a question, or type your request...")

# Display conversation
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input:
    # Attach files to thread if any
    file_ids = []
    if temp_files:
        for file in temp_files:
            try:
                uploaded = openai.files.create(file=file, purpose="fine-tune")
                file_ids.append(uploaded.id)
            except Exception as e:
                st.error(f"Failed to upload {file.name}: {e}")

    # Prepare message parameters
    message_args = {
        "thread_id": st.session_state.thread_id,
        "role": "user",
        "content": user_input
    }
    if file_ids:
        message_args["file_ids"] = file_ids

    # Add user message to thread
    openai.beta.threads.messages.create(**message_args)

    with st.chat_message("user"):
        st.markdown(user_input)

    st.session_state.messages.append({"role": "user", "content": user_input})

    # Run the assistant
    run = openai.beta.threads.runs.create(
        thread_id=st.session_state.thread_id,
        assistant_id=ASSISTANT_ID,
        instructions="You are a multilingual regulatory assistant. Prioritize English summaries and comparisons."
    )

    # Wait for completion
    with st.spinner("Assistant is working..."):
        while True:
            run_status = openai.beta.threads.runs.retrieve(
                thread_id=st.session_state.thread_id,
                run_id=run.id
            )
            if run_status.status in ["completed", "failed"]:
                break

    if run_status.status == "completed":
        messages = openai.beta.threads.messages.list(thread_id=st.session_state.thread_id)
        last_message = messages.data[0].content[0].text.value

        with st.chat_message("assistant"):
            st.markdown(last_message)

        st.session_state.messages.append({"role": "assistant", "content": last_message})
    else:
        st.error("Assistant failed to complete the request.")
