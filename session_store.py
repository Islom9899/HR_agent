import os
import shutil
from typing import Dict, Any
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader


# -------------------------------------------------
# 🔹 문서 로더 (PDF / TXT 지원)
# -------------------------------------------------
def _load_docs(file_path: str):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return PyPDFLoader(file_path).load()
    return TextLoader(file_path, encoding="utf-8").load()


# -------------------------------------------------
# 🔹 세션 생성 (덮어쓰기 허용)
# -------------------------------------------------
def create_or_reset_session(chat_id: str, job_file_path: str):
    """
    [KO] 기존 세션이 존재해도 오류 없이 덮어쓰기 가능하도록 설계.
         Cloud / Local 환경 자동 감지 후 안전하게 저장합니다.
    """

    # Cloud 환경에서는 /mount/temp, 로컬에서는 db 사용
    base_dir = "/mount/temp" if os.path.exists("/mount/temp") else "db"
    persist_dir = os.path.join(base_dir, "sessions", chat_id)
    os.makedirs(persist_dir, exist_ok=True)  # ❗ 디렉토리 미리 생성

    # 기존 컬렉션 파일이 있어도 삭제하지 않음 → 덮어쓰기 가능
    docs = _load_docs(job_file_path)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)
    if not chunks:
        raise ValueError("⚠️ 문서가 비어있거나 청크 분할에 실패했습니다.")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # 🔧 기존 DB가 있어도 새로 추가(업데이트) 가능하도록 설정
    try:
        db = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings,
            collection_name=f"job-{chat_id}",
        )
        db.add_documents(chunks)  # 기존 벡터DB 위에 추가
        db.persist()
        print(f"♻️ 세션 업데이트 완료: {chat_id} (path: {persist_dir})")
    except Exception:
        # 혹시 DB가 손상된 경우 안전하게 재생성
        shutil.rmtree(persist_dir, ignore_errors=True)
        os.makedirs(persist_dir, exist_ok=True)
        Chroma.from_documents(
            chunks,
            embedding=embeddings,
            persist_directory=persist_dir,
            collection_name=f"job-{chat_id}",
        )
        print(f"✅ 새 세션이 재생성되었습니다: {chat_id} (path: {persist_dir})")

    return {"chat_id": chat_id, "persist_dir": persist_dir}


# -------------------------------------------------
# 🔹 세션 종료 (데이터 삭제)
# -------------------------------------------------
def end_session(chat_id: str):
    base_dir = "/mount/temp" if os.path.exists("/mount/temp") else "db"
    persist_dir = os.path.join(base_dir, "sessions", chat_id)
    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir, ignore_errors=True)
        print(f"🧹 세션이 삭제되었습니다: {chat_id}")


# -------------------------------------------------
# 🔹 채용공고 컨텍스트 조회
# -------------------------------------------------
def retrieve_job_context(chat_id: str, query: str = "Evaluate candidate against this job", k: int = 4) -> str:
    base_dir = "/mount/temp" if os.path.exists("/mount/temp") else "db"
    persist_dir = os.path.join(base_dir, "sessions", chat_id)
    if not os.path.isdir(persist_dir):
        return "No job description found in session context."

    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        db = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings,
            collection_name=f"job-{chat_id}",
        )
        docs = db.similarity_search(query, k=k)
        if not docs:
            return "No job description found in session context."
        return "\n\n".join(d.page_content for d in docs if getattr(d, "page_content", "").strip())
    except Exception as e:
        print(f"⚠️ retrieve_job_context 에러: {e}")
        return "No job description found in session context."
