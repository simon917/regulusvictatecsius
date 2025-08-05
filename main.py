import streamlit as st
import openai
from datetime import datetime

# Load secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]
ASSISTANT_ID = st.secrets["ASSISTANT_ID"]
VECTORSTORE_ID = st.secrets["VECTORSTORE_ID"]

# UI setup
st.set_page_config(page_title="Regulatory Assistant UI", layout="wide")
st.title("📄 AI Regulatory Document Assistant")

# Thread setup
if "thread_id" not in st.session_state:
    thread = openai.beta.threads.create()
    st.session_state.thread_id = thread.id

# List vector store files
st.sidebar.markdown("---")
st.sidebar.subheader("📄 Files in Vector Store")
try:
    vector_files = openai.vector_stores.files.list(vector_store_id=VECTORSTORE_ID).data
    if vector_files:
        for f in vector_files:
            file = openai.files.retrieve(f.file_id)
            timestamp = datetime.fromtimestamp(file.created_at).strftime("%Y-%m-%d %H:%M")
            url = f"https://api.openai.com/v1/files/{file.id}/content"
            st.sidebar.markdown(f"[{file.filename} — {timestamp}]({url})")
    else:
        st.sidebar.info("Vector store is empty.")
        st.sidebar.info("Assistant has no linked files.")
except Exception as e:
    st.sidebar.error(f"Error retrieving assistant files: {e}")
    st.sidebar.info("No files uploaded in this session.")

# Chat section
st.subheader("💬 Chat with your Assistant")
temp_files = st.file_uploader("Attach PDFs (one-off use)", type=["pdf"], accept_multiple_files=True)
user_input = st.chat_input("Ask a question or describe your task...")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input:
    file_ids = []
    if temp_files:
        for file in temp_files:
            try:
                uploaded = openai.files.create(file=file, purpose="assistants")
                file_ids.append(uploaded.id)
            except Exception as e:
                st.error(f"Failed to upload: {e}")

    message_args = {
        "thread_id": st.session_state.thread_id,
        "role": "user",
        "content": user_input
    }
    if file_ids:
        message_args["file_ids"] = file_ids

    openai.beta.threads.messages.create(**message_args)
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    run = openai.beta.threads.runs.create(
        thread_id=st.session_state.thread_id,
        assistant_id=ASSISTANT_ID,
        instructions="You are a multilingual regulatory assistant. Prioritize English summaries and comparisons."
    )

    with st.spinner("Assistant is thinking..."):
        while True:
            status = openai.beta.threads.runs.retrieve(
                thread_id=st.session_state.thread_id,
                run_id=run.id
            )
            if status.status in ["completed", "failed"]:
                break

    if status.status == "completed":
        messages = openai.beta.threads.messages.list(thread_id=st.session_state.thread_id)
        reply = messages.data[0].content[0].text.value
        st.chat_message("assistant").markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})
    else:
        st.error("Assistant failed to respond.")
