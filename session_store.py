from pathlib import Path
from typing import List
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# 세션별로 채용 공고 문맥을 벡터DB(Chroma)에 저장/조회
ROOT = Path(__file__).resolve().parents[1]
SESS_DB_ROOT = ROOT / "db" / "sessions"
SESS_DB_ROOT.mkdir(parents=True, exist_ok=True)

def _load_docs(file_path: str):
    """
    파일 확장자에 따라 PDF/Text 로더 선택
    """
    p = Path(file_path)
    if p.suffix.lower() == ".pdf":
        return PyPDFLoader(str(p)).load()
    return TextLoader(str(p), encoding="utf-8").load()

def create_or_reset_session(chat_id: str, job_file_path: str):
    """
    세션 생성/리셋
    1) 채용 공고 파일을 문서 조각으로 분할
    2) 임베딩 생성
    3) 세션별 Chroma 벡터스토어에 저장(지속)
    """
    docs = _load_docs(job_file_path)

    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    for ch in chunks:
        ch.metadata = {"chat_id": chat_id}

    persist_dir = str(SESS_DB_ROOT / chat_id)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    # 최신 Chroma에서는 .persist() 호출 없이도 자동 지속 저장
    Chroma.from_documents(chunks, embedding=embeddings, persist_directory=None)
    return {"chat_id": chat_id, "persist_dir": persist_dir}

def retrieve_job_context(chat_id: str, query: str = "Evaluate candidate against this job"):
    """
    세션 벡터DB에서 관련 문맥 검색 후 상위 k개를 합쳐 반환
    """
    persist_dir = str(SESS_DB_ROOT / chat_id)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    db = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    hits = db.similarity_search(query, k=4)
    return "\n\n".join([h.page_content for h in hits])

def end_session(chat_id: str):
    """
    세션 종료 시 디스크 정리(선택)
    """
    import shutil
    dirp = SESS_DB_ROOT / chat_id
    if dirp.exists():
        shutil.rmtree(dirp, ignore_errors=True)




