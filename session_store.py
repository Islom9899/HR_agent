from pathlib import Path
import shutil
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


ROOT = Path(__file__).resolve().parents[1]
SESS_DB_ROOT = ROOT / "db" / "sessions"
SESS_DB_ROOT.mkdir(parents=True, exist_ok=True)


def _load_docs(file_path: str):
    """PDF/TXT 파일 로더"""
    p = Path(file_path)
    if p.suffix.lower() == ".pdf":
        return PyPDFLoader(str(p)).load()
    return TextLoader(str(p), encoding="utf-8").load()


def create_or_reset_session(chat_id: str, job_file_path: str):
    """
    세션 생성 / 리셋
    ----------------------------------
    채용 공고를 문서 조각으로 분할 → 임베딩 생성 → FAISS에 저장
    """
    docs = _load_docs(job_file_path)
    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    for ch in chunks:
        ch.metadata = {"chat_id": chat_id}

    persist_dir = str(SESS_DB_ROOT / chat_id)
    if Path(persist_dir).exists():
        shutil.rmtree(persist_dir, ignore_errors=True)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # 🔹 FAISS로 전환 — 로컬 서버 불필요
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(persist_dir)

    return {"chat_id": chat_id, "persist_dir": persist_dir}


def retrieve_job_context(chat_id: str, query: str = "Evaluate candidate against this job"):
    """
    🔍 채용 공고 문맥 검색 (FAISS)
    """
    persist_dir = str(SESS_DB_ROOT / chat_id)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # allow_dangerous_deserialization=True → FAISS 안전하게 불러오기
    db = FAISS.load_local(persist_dir, embeddings, allow_dangerous_deserialization=True)
    hits = db.similarity_search(query, k=4)
    return "\n\n".join([h.page_content for h in hits])


def end_session(chat_id: str):
    """
    세션 종료 시 디스크 및 메모리 정리
    """
    import streamlit as st
    from streamlit.runtime.caching import clear_cache

    try:
        dirp = SESS_DB_ROOT / chat_id
        if dirp.exists():
            shutil.rmtree(dirp, ignore_errors=True)
    except Exception:
        pass

    try:
        st.session_state.clear()
        clear_cache()
    except Exception:
        pass
