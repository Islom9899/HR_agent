from pydantic import BaseModel, Field
from typing import List, Optional

class ResumeExtract(BaseModel):
    """
    이력서에서 추출된 구조화 정보 모델
    - 채용 스크리닝 단계에서 일관성 있고 타입 안정적인 필드를 제공
    """
    name: Optional[str] = Field(None, description="지원자 성명(있으면)")
    summary: Optional[str] = Field(None, description="개요/프로필 요약")
    years_experience: Optional[float] = Field(
        None, description="총 경력 연차(가능하면 float)"
    )
    skills: List[str] = Field(
        default_factory=list, description="정규화된 기술 스택(소문자 리스트)"
    )
    education: Optional[str] = Field(
        None, description="최종 학력 또는 핵심 학력 요약"
    )
    recent_companies: List[str] = Field(
        default_factory=list, description="최근 재직 회사 목록"
    )
    projects: List[str] = Field(
        default_factory=list, description="핵심 프로젝트 간단 리스트"
    )

class ScoreBreakdown(BaseModel):
    """
    평가 점수 상세
    - 모든 점수는 0~100 범위
    - UI에서는 total 키를 사용하므로 total 필드를 명시
    """
    skill_match: int = Field(description="기술 스킬 적합도 0-100")
    experience_match: int = Field(description="경력 적합도 0-100")
    education_match: int = Field(description="학력 적합도 0-100")
    total: int = Field(description="종합 점수 0-100")

class HRDecision(BaseModel):
    """
    최종 HR 판단 결과
    - decision: PASS 또는 REJECT
    - reasons: 판단 근거
    - improvements: 후보자 개선 제안
    - score: 점수 상세(ScoreBreakdown)
    """
    decision: str = Field(description="PASS 또는 REJECT")
    reasons: List[str] = Field(description="판단 근거(간결하게)")
    improvements: List[str] = Field(description="개선 제안")
    score: ScoreBreakdown
