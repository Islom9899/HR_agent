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
    [KO] 파일 확장자에 따라 문서를 불러옵니다.
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return PyPDFLoader(file_path).load()
    return TextLoader(file_path, encoding="utf-8").load()


# ---------------------------------------
# 🔹 세션 생성 (기존 세션 삭제 후 새로 생성)
# ---------------------------------------
def create_or_reset_session(chat_id: str, job_file_path: str):
    """
    [KO] 기존 세션 폴더가 존재하면 삭제 후 새로 생성합니다.
         로컬 및 Streamlit Cloud 환경에서 모두 작동하도록 설계되었습니다.
    """

    # 📂 세션 데이터 저장 경로
    persist_dir = os.path.join("db", "sessions", chat_id)

    # 🔥 기존 세션 폴더 삭제
    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir, ignore_errors=True)
    os.makedirs(persist_dir, exist_ok=True)

    # 📄 문서 로드
    docs = _load_docs(job_file_path)

    # ✂️ 텍스트 분할
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)

    if not chunks:
        raise ValueError("⚠️ 문서가 비어있거나 청크 분할에 실패했습니다.")

    # 🧠 Embedding 생성
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # 💾 ChromaDB 저장
    Chroma.from_documents(
        chunks,
        embedding=embeddings,
        persist_directory=persist_dir
    )

    print(f"✅ 새 세션이 생성되었습니다: {chat_id} (path: {persist_dir})")
    return {"chat_id": chat_id, "persist_dir": persist_dir}


# ---------------------------------------
# 🔹 세션 종료 (데이터 삭제)
# ---------------------------------------
def end_session(chat_id: str):
    """
    [KO] 세션 종료 시 저장된 데이터를 삭제합니다.
    """
    persist_dir = os.path.join("db", "sessions", chat_id)

    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir, ignore_errors=True)
        print(f"🧹 세션이 삭제되었습니다: {chat_id}")


# ---------------------------------------
# 🔹 채용공고 컨텍스트 조회
# ---------------------------------------
def retrieve_job_context(chat_id: str, query: str = "Evaluate candidate against this job", k: int = 4) -> str:
    """
    [KO] 세션에 저장된 채용공고 문맥을 조회합니다.
    """
    persist_dir = os.path.join("db", "sessions", chat_id)

    if not os.path.isdir(persist_dir):
        return "No job description found in session context."

    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        db = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings,
        )
        docs = db.similarity_search(query, k=k)
        if not docs:
            return "No job description found in session context."
        return "\n\n".join(d.page_content for d in docs if getattr(d, "page_content", "").strip())
    except Exception:
        return "No job description found in session context."
