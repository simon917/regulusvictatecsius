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

# Vector store upload
st.sidebar.header("📚 Add to Knowledge Base")
persist_files = st.sidebar.file_uploader("Upload PDFs to vector store", type=["pdf"], accept_multiple_files=True)
if persist_files:
    for file in persist_files:
        try:
            uploaded = openai.files.create(file=file, purpose="assistants")
            assistant = openai.beta.assistants.retrieve(ASSISTANT_ID)
            tool_resources = assistant.model_dump().get("tool_resources", {})
            current_files = tool_resources.get("file_search", {}).get("file_ids", [])

            openai.beta.assistants.update(
                assistant_id=ASSISTANT_ID,
                name=assistant.name,
                instructions=assistant.instructions,
                tools=assistant.tools,
                model=assistant.model,
                tool_resources={
                    "file_search": {
                        "vector_store_ids": [VECTORSTORE_ID],
                        "file_ids": list(set(current_files + [uploaded.id]))
                    }
                }
            )
            st.sidebar.success(f"Uploaded: {file.name}")
        except Exception as e:
            st.sidebar.error(f"Upload failed: {e}")

# List vector store files
st.sidebar.markdown("---")
st.sidebar.subheader("📄 Files in Vector Store")
try:
    assistant = openai.beta.assistants.retrieve(ASSISTANT_ID)
    file_ids = assistant.file_ids if hasattr(assistant, "file_ids") else []
    vector_files = [openai.files.retrieve(fid) for fid in file_ids]
    if vector_files:
        for f in vector_files:
            col1, col2 = st.sidebar.columns([4, 1])
            timestamp = datetime.fromtimestamp(f.created_at).strftime("%Y-%m-%d %H:%M")
            url = f"https://api.openai.com/v1/files/{f.id}/content"
            with col1:
                st.markdown(f"[{f.filename} — {timestamp}]({url})")
            with col2:
                if st.button("🗑️", key=f"del_{f.id}"):
                    if st.sidebar.checkbox(f"Confirm delete: {f.filename}", key=f"cfm_{f.id}"):
                        try:
                            openai.files.delete(f.id)
                            st.experimental_rerun()
                        except Exception as e:
                            st.sidebar.error(f"Error deleting: {e}")
    else:
        st.sidebar.info("No files in vector store yet.")
except Exception as e:
    st.sidebar.error(f"Error loading files: {e}")

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
