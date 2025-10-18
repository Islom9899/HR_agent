import os
import shutil
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader


def _load_docs(file_path: str):
    """
    [KO] 파일 확장자에 따라 문서 로더 선택
         - PDF 파일: PyPDFLoader 사용
         - TXT 파일: TextLoader 사용
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return PyPDFLoader(file_path).load()
    return TextLoader(file_path, encoding="utf-8").load()


def create_or_reset_session(chat_id: str, job_file_path: str):
    """
    [KO] 기존 세션이 존재하면 삭제 후 새로 생성하는 함수

    - 동일한 chat_id를 사용할 때 이전 세션의 벡터DB가 남아 있으면
      Chroma가 충돌을 일으킬 수 있으므로, 기존 세션 폴더를 완전히 제거합니다.
    - 새로운 세션 디렉토리를 만든 뒤, 채용 공고 텍스트를 로드하고
      Embedding을 생성하여 ChromaDB에 저장합니다.

    Args:
        chat_id (str): 세션 ID (예: "hr_chat_001")
        job_file_path (str): 채용 공고 텍스트 또는 PDF 파일 경로

    Returns:
        dict: 세션 정보 (chat_id, persist_dir)
    """

    # ✅ 세션 저장 경로 설정
    persist_dir = os.path.join("db", "sessions", chat_id)

    # 🔥 기존 세션 폴더가 존재할 경우 삭제 (reset)
    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir, ignore_errors=True)
    os.makedirs(persist_dir, exist_ok=True)

    # 📄 채용 공고 문서 로드
    docs = _load_docs(job_file_path)

    # ✂️ 문서를 청크 단위로 분할
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)

    if not chunks:
        raise ValueError("⚠️ 문서가 비어 있거나 청크 분할에 실패했습니다.")

    # 🧠 OpenAI Embedding 모델 초기화
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # 🧩 Chroma 벡터 데이터베이스 생성 (새 세션용)
    Chroma.from_documents(
        chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_name=f"job-{chat_id}",
    )

    print(f"✅ 새 세션이 생성되었습니다: {chat_id}")
    return {"chat_id": chat_id, "persist_dir": persist_dir}


def end_session(chat_id: str):
    """
    [KO] 세션 종료 시 저장된 데이터 삭제 (선택적 사용)
         - 세션이 종료되면 불필요한 Chroma 폴더를 제거합니다.
    """
    persist_dir = os.path.join("db", "sessions", chat_id)
    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir, ignore_errors=True)
        print(f"🧹 세션이 삭제되었습니다: {chat_id}")
