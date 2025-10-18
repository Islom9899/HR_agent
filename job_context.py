import os
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings


# -------------------------------------------------
# 🔹 채용공고 문맥 불러오기
# -------------------------------------------------
def job_context_node(chat_id: str, query: str = "지원자 이력서를 이 채용공고에 맞춰 평가하세요.", k: int = 4) -> str:
    """
    [KO] 세션에 저장된 채용공고 문맥(Job Context)을 검색하여 반환합니다.

    Args:
        chat_id (str): 세션 ID (예: "hr_chat_001")
        query (str): 검색 쿼리 (기본값은 HR 평가용)
        k (int): 검색할 문서의 개수 (유사도 상위 k개)

    Returns:
        str: 채용공고의 핵심 문맥 (없을 경우 기본 안내 메시지)
    """

    # 🌐 환경에 따라 세션 디렉토리 경로 설정
    base_dir = "/mount/temp" if os.path.exists("/mount/temp") else "db"
    persist_dir = os.path.join(base_dir, "sessions", chat_id)

    # 📁 세션 폴더 확인
    if not os.path.isdir(persist_dir):
        print(f"⚠️ 세션 데이터가 존재하지 않습니다: {persist_dir}")
        return "No job description found in session context."

    try:
        # 🧠 Embedding 모델 (session_store.py와 동일)
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

        # 💾 ChromaDB에서 데이터 로드
        db = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings,
            collection_name=f"job-{chat_id}",
        )

        # 🔍 유사도 검색 (job context)
        docs = db.similarity_search(query, k=k)

        # 📄 결과 텍스트 병합
        if not docs:
            print(f"⚠️ 채용공고 문맥을 찾을 수 없습니다. (chat_id: {chat_id})")
            return "No job description found in session context."

        context_text = "\n\n".join(
            d.page_content for d in docs if getattr(d, "page_content", "").strip()
        )

        print(f"✅ 채용공고 문맥 로드 완료 (chat_id: {chat_id})")
        return context_text

    except Exception as e:
        print(f"⚠️ job_context_node 오류 발생: {e}")
        return "No job description found in session context."
