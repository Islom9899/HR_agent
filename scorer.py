from typing import Dict
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from schemas import ResumeExtract, HRDecision

# 평가 시스템 지침
# - must-have 스킬 충족 + total 점수(0~100)가 threshold 이상이면 PASS
scorer_system = (
    "당신은 HR 스크리닝 보조자입니다. 후보자의 추출 이력서를 채용 요구사항과 비교해 "
    "기술/경력/학력을 각각 0~100으로 점수화하고, 'total' 종합 점수를 산출하세요. "
    "must-have 스킬 충족 여부를 엄격히 판단하고, total >= threshold 이고 must-have가 충분히 충족될 때만 PASS를 내리세요."
)

# 사람 프롬프트
scorer_prompt = ChatPromptTemplate.from_messages([
    ("system", scorer_system),
    (
        "human",
        "채용 공고:\n```\n{job_description}\n```\n\n"
        "최소 경력 연차: {min_years}\n"
        "Must-have 스킬: {must_have_skills}\n"
        "Nice-to-have 스킬: {nice_to_have_skills}\n"
        "PASS 임계값(종합 점수): {threshold}\n\n"
        "추출된 이력서(JSON):\n{extracted_json}\n\n"
        "반환 형식: PASS 또는 REJECT, 사유(reasons), 개선점(improvements), 점수 상세(score: skill_match, experience_match, education_match, total)."
    ),
])

_llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

# HRDecision 스키마에 맞춘 구조화 출력
scorer_chain = scorer_prompt | _llm.with_structured_output(HRDecision)

def scorer_node(state: Dict) -> Dict:
    """
    스코어링 노드
    입력 state:
      - extracted: ResumeExtract
      - job_description, min_years, must_have_skills, nice_to_have_skills, threshold
    출력:
      - decision, reasons, improvements, score(dict; total 포함)
    """
    extracted: ResumeExtract = state["extracted"]
    threshold: int = state.get("threshold", 70)

    result: HRDecision = scorer_chain.invoke({
        "job_description": state["job_description"],
        "min_years": state["min_years"],
        "must_have_skills": state["must_have_skills"],
        "nice_to_have_skills": state["nice_to_have_skills"],
        "threshold": threshold,
        "extracted_json": extracted.model_dump_json(indent=2),
    })

    return {
        "decision": result.decision,
        "reasons": result.reasons,
        "improvements": result.improvements,
        "score": result.score.model_dump(),  # total 키 포함
    }
