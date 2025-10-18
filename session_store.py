import os
import shutil
import chromadb
from typing import Dict, Any
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader


# -----------------------------
# 🔹 문서 로더 (PDF / TXT 지원)
# -----------------------------
def _load_docs(file_path: str):
    """
    파일 확장자에 따라 문서를 불러옵니다.
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return PyPDFLoader(file_path).load()
    return TextLoader(file_path, encoding="utf-8").load()


# ---------------------------------------
# 🔹 세션 생성 (Ephemeral 모드)
# ---------------------------------------
def create_or_reset_session(chat_id: str, job_file_path: str):
    """
    기존 세션과 상관없이 항상 새 Chroma 세션(메모리 기반)을 생성합니다.
    Streamlit Cloud에서도 안전하게 작동합니다.
    """

    # 🔹 채용 공고 파일 로드
    docs = _load_docs(job_file_path)

    # 🔹 텍스트 분할
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)
    if not chunks:
        raise ValueError("⚠️ 문서가 비어있거나 청크 분할에 실패했습니다.")

    # 🔹 임베딩 초기화
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # 🔹 Ephemeral(메모리 전용) 클라이언트 생성
    client = chromadb.EphemeralClient()

    # 🔹 ChromaDB 생성 (메모리 모드)
    db = Chroma.from_documents(
        chunks,
        embedding=embeddings,
        client=client,
        collection_name=f"job-{chat_id}",
    )

    print(f"✅ Ephemeral 세션이 생성되었습니다: {chat_id}")
    # 세션 상태 반환 (persist_dir이 없음)
    return {"chat_id": chat_id, "persist_dir": None, "client": client, "db": db}


# ---------------------------------------
# 🔹 세션 종료 (메모리 모드이므로 단순 로그)
# ---------------------------------------
def end_session(chat_id: str):
    """
    메모리 기반 세션 종료 시 별도의 삭제는 필요 없습니다.
    """
    print(f"🧹 세션 종료: {chat_id}")


# ---------------------------------------
# 🔹 채용공고 컨텍스트 조회
# ---------------------------------------
def retrieve_job_context(chat_id: str, query: str = "Evaluate candidate against this job", k: int = 4) -> str:
    """
    메모리 세션에서는 DB 파일이 존재하지 않기 때문에,
    실제 컨텍스트는 graph 실행 중 상태(state)로 전달되어야 합니다.
    여기서는 단순 placeholder 문자열을 반환합니다.
    """
    return f"[INFO] 세션 {chat_id}의 채용공고 문맥이 로드되었습니다."
