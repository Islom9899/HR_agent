from pathlib import Path
import shutil
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import chromadb
from chromadb.config import Settings


# ==========================================================
#  세션별 채용 공고 문맥(Vector DB) 관리 모듈
# ----------------------------------------------------------
# - 각 세션(chat_id)마다 별도의 Chroma DB를 생성 및 저장
# - Streamlit Cloud 환경 호환 (telemetry 비활성화)
# - 세션 종료 시 디스크 및 메모리 자동 정리
# ==========================================================

ROOT = Path(__file__).resolve().parents[1]
SESS_DB_ROOT = ROOT / "db" / "sessions"
SESS_DB_ROOT.mkdir(parents=True, exist_ok=True)


def _load_docs(file_path: str):
    """
    📄 파일 로더
    -----------------
    파일 확장자에 따라 적절한 로더 선택:
      - PDF → PyPDFLoader
      - TXT → TextLoader
    """
    p = Path(file_path)
    if p.suffix.lower() == ".pdf":
        return PyPDFLoader(str(p)).load()
    return TextLoader(str(p), encoding="utf-8").load()


def create_or_reset_session(chat_id: str, job_file_path: str):
    """
    세션 생성 / 리셋
    ----------------------------------
    채용 공고 파일을 불러와 문서 조각으로 분할
    OpenAI Embedding 생성
    세션별 Chroma VectorStore에 저장 (자동 지속)
    """
    docs = _load_docs(job_file_path)
    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    for ch in chunks:
        ch.metadata = {"chat_id": chat_id}

    persist_dir = str(SESS_DB_ROOT / chat_id)

    # 기존 세션 디렉토리 제거 (깨진 DB 방지)
    if Path(persist_dir).exists():
        shutil.rmtree(persist_dir, ignore_errors=True)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # 안정 버전용 Chroma 설정
    client_settings = Settings(
        anonymized_telemetry=False,
        is_persistent=True,
        allow_reset=True,
        persist_directory=persist_dir
    )

    # 문서로부터 벡터 DB 생성
    Chroma.from_documents(
        chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
        client_settings=client_settings
    )

    return {"chat_id": chat_id, "persist_dir": persist_dir}


def retrieve_job_context(chat_id: str, query: str = "Evaluate candidate against this job"):
    """
    🔍 채용 공고 문맥 검색
    ----------------------------------
    세션별 Chroma DB에서 유사한 문서를 검색하여
    상위 k개 조각을 합쳐 반환한다.
    """
    persist_dir = str(SESS_DB_ROOT / chat_id)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    db = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    hits = db.similarity_search(query, k=4)
    return "\n\n".join([h.page_content for h in hits])


def end_session(chat_id: str):
    """
    세션 종료 시 디스크 및 메모리 정리
    ----------------------------------
    - Chroma 클라이언트 종료
    - 세션 디렉토리 삭제
    - Streamlit 캐시 및 상태 초기화
    """
    import streamlit as st
    from streamlit.runtime.caching import clear_cache

    persist_dir = str(SESS_DB_ROOT / chat_id)
    try:
        # 1️⃣ 메모리에 남은 Chroma 인스턴스 정리
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        db = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
        db.delete_collection()
        del db
    except Exception:
        pass

    try:
        # 2️⃣ 디스크 삭제
        dirp = SESS_DB_ROOT / chat_id
        if dirp.exists():
            shutil.rmtree(dirp, ignore_errors=True)
    except Exception:
        pass

    try:
        # 3️⃣ Streamlit 세션 상태 및 캐시 초기화
        st.session_state.clear()
        clear_cache()
    except Exception:
        pass
