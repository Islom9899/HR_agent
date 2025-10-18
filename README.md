# 🧠 HR AI Screening Agent   
> AI-driven resume screening system built with **LangGraph**, **LangChain**, **OpenAI GPT-4o-mini**, and **Streamlit**.  
> 인공지능 기반 이력서 자동 평가 시스템 — 채용 담당자의 스크리닝 과정을 완전히 자동화합니다.

---

## 🌟 Overview | 프로젝트 개요

**HR AI Screening Agent**는 지원자의 이력서를 채용 공고와 자동으로 비교하여,  
경험·기술·학력 적합도를 정량적으로 평가하는 **AI HR 스크리닝 시스템**입니다.  

이 프로젝트는 **LangGraph 워크플로우**, **GPT-4o-mini**, **ChromaDB**,  
그리고 **Streamlit**을 이용해 실제 HR 프로세스를 자동화한 형태로 제작되었습니다.  

---

## 🚀 Key Features | 주요 기능

| 기능 | 설명 |
|------|------|
| 🧩 **LangGraph Workflow** | `defaults → job_context → loader → extractor → scorer` 구조로 구성된 완전한 HR 그래프 파이프라인 |
| 🤖 **AI Resume Extraction** | GPT-4o-mini를 활용해 이력서의 핵심 정보를 구조화(name, skills, experience 등) |
| 📊 **Automated Scoring** | Must-have / Nice-to-have 스킬, 경력 연차, 전공 적합도를 기반으로 점수화 (0–100) |
| 🧠 **Session-based Context** | 채용 공고 정보를 ChromaDB에 저장하고, 세션 단위로 재사용 |
| 💼 **Streamlit Interface** | 단일/배치 평가, 결과 시각화, CSV 다운로드 기능 제공 |
| 🌍 **Multilingual Ready** | 영어 / 한국어 프롬프트 모두 지원 (추가 언어 확장 가능) |

---

## 🏗️ Architecture | 아키텍처

HR Agent
│
├── app.py → Streamlit UI
├── graph_builder.py → LangGraph 파이프라인 구성
├── extractor.py → 이력서 텍스트 → 구조화 데이터 추출
├── scorer.py → 점수 계산 및 PASS/REJECT 결정
├── session_store.py → ChromaDB 세션 저장소
├── job_context.py → 채용 공고 문맥 노드
├── loader.py → 이력서 로딩 (PDF/TXT)
└── schemas.py → Pydantic 스키마 정의

---

## 🔁 Workflow Diagram | 처리 흐름도

START
↓
defaults (기본값 세팅)
↓
job_context (채용 공고 로드)
↓
loader (이력서 텍스트 로드)
↓
extractor (AI 기반 정보 추출)
↓
scorer (점수화 + PASS/REJECT)
↓
END

---

## 💡 Example | 평가 예시

**Job Description (채용 공고):**  
> Backend Developer (Python, FastAPI, SQL, Pandas)

**Candidate Resume (이력서):**  
> 2년 경력, Python + SQL 능숙, FastAPI 기반 REST API 개발 경험  
> Pandas로 데이터 파이프라인 구축 경험

**Result →** ✅ PASS (Total: 95)

---

## ⚙️ Installation | 설치 방법

1️⃣ **프로젝트 복제 / Clone the repository**
```bash
git clone https://github.com/Islom9899/HR_agent.git
cd HR_agent

---
## requirements.txt
pip install -r requirements.txt
---
## .env
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxx
---
## 앱 실행
streamlit run app.py
---
## 🧠 Technologies | 사용 기술 스택

| Category            | Stack                                    |
| ------------------- | ---------------------------------------- |
| **LLM & AI**        | OpenAI GPT-4o-mini, LangChain, LangGraph |
| **Frontend**        | Streamlit                                |
| **Vector DB**       | ChromaDB                                 |
| **Backend / Logic** | Python 3.10+, Pydantic                   |
| **Infra & Utils**   | dotenv, pandas, OpenAI Embeddings        |

---
## 📈 Score Logic | 평가 로직 요약

| 평가 항목            | 설명                              | 비중  |
| ---------------- | ------------------------------- | --- |
| Skill Match      | Must-have / Nice-to-have 스킬 일치도 | 40% |
| Experience Match | 경력 연차 적합도                       | 30% |
| Education Match  | 전공 일치도                          | 30% |
| **Threshold**    | **기본값 70 (합격 기준)**              | —   |

---
## 🧰 Folder Structure | 폴더 구조
📦 HR_AI_Screening_Agent
├── app.py
├── extractor.py
├── scorer.py
├── session_store.py
├── job_context.py
├── loader.py
├── graph_builder.py
├── schemas.py
└── README.md


