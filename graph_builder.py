from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

from job_context import job_context_node  # 세션별 채용공고 문맥 구성 노드
from loader import loader_node            # 이력서 파일 로더 노드
from extractor import extractor_node      # 텍스트 -> 구조화 추출 노드
from scorer import scorer_node            # 점수화/판단 노드

class HRState(TypedDict, total=False):
    """
    그래프 상태 타입
    - 각 노드 간 전달되는 키를 정의
    """
    chat_id: str
    resume_path: str
    job_description: str
    min_years: float
    must_have_skills: list[str]
    nice_to_have_skills: list[str]
    threshold: int
    resume_text: str
    extracted: dict
    decision: str
    reasons: list[str]
    improvements: list[str]
    score: dict

def build_graph():
    """
    LangGraph 빌드:
    START -> defaults -> job_context -> loader -> extractor -> scorer -> END
    """
    g = StateGraph(HRState)

    # --- 노드 등록 ---
    g.add_node("defaults", _defaults_node)
    g.add_node("job_context", job_context_node)
    g.add_node("loader", loader_node)
    g.add_node("extractor", extractor_node)
    g.add_node("scorer", scorer_node)

    # --- 엣지(흐름) ---
    g.add_edge(START, "defaults")
    g.add_edge("defaults", "job_context")
    g.add_edge("job_context", "loader")
    g.add_edge("loader", "extractor")
    g.add_edge("extractor", "scorer")
    g.add_edge("scorer", END)

    return g.compile()

def _defaults_node(state: HRState) -> HRState:
    """
    기본값 설정 노드
    - 그래프 초기에 누락된 설정을 안전하게 채움
    """
    state.setdefault("min_years", 1.0)
    state.setdefault("threshold", 70)
    state.setdefault("must_have_skills", ["python", "sql"])
    state.setdefault("nice_to_have_skills", ["pandas", "fastapi"])
    return state
