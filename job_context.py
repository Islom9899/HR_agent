from typing import Dict
from session_store import retrieve_job_context

def job_context_node(state: Dict) -> Dict:
    """
    채용 공고 문맥 노드
    - 세션 ID(chat_id)를 기반으로 저장된 벡터 DB(Chroma)에서
      채용 공고 문맥을 검색하여 job_description 필드로 반환한다.
    
    입력:
        state["chat_id"]: 세션 ID
    출력:
        {"job_description": str}
    """
    chat_id = state.get("chat_id")
    if not chat_id:
        raise ValueError("chat_id가 누락되었습니다. 세션이 생성되지 않았습니다.")

    # 세션 DB에서 채용 공고 문맥 검색
    job_desc = retrieve_job_context(chat_id)
    if not job_desc or len(job_desc.strip()) == 0:
        job_desc = "No job description found in session context."

    return {"job_description": job_desc}
