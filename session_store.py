from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
import os

def create_or_reset_session(chat_id: str, job_file_path: str):
    persist_dir = os.path.join("db", "sessions", chat_id)
    os.makedirs(persist_dir, exist_ok=True)

    with open(job_file_path, "r", encoding="utf-8") as f:
        job_text = f.read()

    if not job_text.strip():
        raise ValueError("Job file is empty.")

    # Embedding model
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Text splitter
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.create_documents([job_text])

    if not chunks:
        raise ValueError("No valid text chunks were created from job description.")

    # 🧠 Use in-memory Chroma (no persist)
    db = Chroma.from_documents(chunks, embedding=embeddings, persist_directory=None)

    print(f"✅ Session created for chat_id={chat_id}")
    return db
