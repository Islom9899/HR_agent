from typing import Dict
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from schemas import ResumeExtract
from dotenv import load_dotenv

load_dotenv()

# 이력서 텍스트에서 구조화된 필드를 추출하기 위한 프롬프트
extract_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "당신은 HR 전문가입니다. 이력서 텍스트에서 구조화된 정보를 정확히 추출하세요."
    ),
    (
        "human",
        "이력서 텍스트:\n'''{resume_text}'''\n\n"
        "다음 필드로 구조화하여 반환하세요:\n"
        "- name (문자열)\n"
        "- summary (문자열)\n"
        "- years_experience (가능하면 float)\n"
        "- skills (소문자 문자열 리스트, 정규화)\n"
        "- education (짧은 문자열)\n"
        "- recent_companies (문자열 리스트)\n"
        "- projects (문자열 리스트)"
    ),
])

llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

# 구조화된 출력(ResumeExtract) 강제
extract_chain = extract_prompt | llm.with_structured_output(ResumeExtract)

def extractor_node(state: Dict) -> Dict:
    """
    이력서 추출 노드
    입력:
      - state['resume_text']: 원본 이력서 텍스트
    출력:
      - {'extracted': ResumeExtract}
    """
    extracted: ResumeExtract = extract_chain.invoke({'resume_text': state['resume_text']})
    return {"extracted": extracted}

