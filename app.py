import os
import io
import time
import json
import shutil
from pathlib import Path
from typing import Dict, Any, List
import streamlit as st
import pandas as pd
from session_store import create_or_reset_session, end_session
from graph_builder import build_graph
from dotenv import load_dotenv

# ------------------- ENVIRONMENT -------------------
load_dotenv()

# Streamlit 기본 설정
st.set_page_config(
    page_title="HR AI Screening Agent",
    page_icon="🧠",
    layout="wide",
)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
REQ_DIR = DATA_DIR / "requirements" / "sessions"
RES_DIR = DATA_DIR / "resumes"
OUT_DIR = DATA_DIR / "outputs"

for d in [REQ_DIR, RES_DIR, OUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ------------------- HELPER FUNCTIONS -------------------
def save_upload(file, dst_dir: Path) -> str:
    """업로드 파일을 안전한 이름으로 저장하고 경로 반환"""
    dst_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(file.name).suffix.lower()
    safe_name = f"{int(time.time())}_{Path(file.name).stem}{suffix}"
    path = dst_dir / safe_name
    path.write_bytes(file.read())
    return str(path)

def decision_badge(decision: str) -> str:
    """PASS/REJECT 뱃지 표시용 텍스트"""
    if not decision:
        return "⚪️ UNKNOWN"
    d = decision.strip().upper()
    if d.startswith("PASS") or d == "ACCEPT":
        return "🟢 PASS"
    if d.startswith("REJECT") or d == "FAIL":
        return "🔴 REJECT"
    return f"⚪️ {d}"

def pretty_score(score: Dict[str, Any]) -> pd.DataFrame:
    """점수 딕셔너리를 테이블로 변환 (정수 캐스팅 시도)"""
    if not isinstance(score, dict):
        return pd.DataFrame()
    rows = []
    for k, v in score.items():
        try:
            rows.append({"metric": k, "score": int(v)})
        except Exception:
            rows.append({"metric": k, "score": v})
    return pd.DataFrame(rows)

@st.cache_resource(show_spinner=False)
def ensure_graph():
    """LangGraph 빌드 및 캐시"""
    return build_graph()

# ------------------- SIDEBAR -------------------
with st.sidebar:
    st.header("🔐 세션 관리")
    chat_id = st.text_input("Chat ID (세션)", value=st.session_state.get("chat_id", "hr_chat_001"))
    job_file = st.file_uploader("채용 공고 (.pdf/.txt) — 1회 업로드", type=["pdf", "txt"])

    colA, colB = st.columns(2)
    with colA:
        if st.button("🚀 세션 생성/갱신", use_container_width=True, disabled=not chat_id or not job_file):
            with st.spinner("세션 초기화 중..."):
                job_path = save_upload(job_file, REQ_DIR / chat_id)
                create_or_reset_session(chat_id=chat_id, job_file_path=job_path)
                st.session_state["chat_id"] = chat_id
            st.success(f"세션 생성됨: {chat_id}")

    with colB:
        if st.button("🧹 세션 종료", use_container_width=True, disabled=not chat_id):
            with st.spinner("세션 정리 중..."):
                try:
                    end_session(chat_id)
                except Exception:
                    pass
                shutil.rmtree(REQ_DIR / chat_id, ignore_errors=True)
                st.session_state.pop("chat_id", None)
                st.cache_resource.clear()  # ✅ Cache tozalash (eng muhim qator)
            st.success("세션 종료 및 정리 완료")

    st.markdown("---")
    st.caption("⚠️ 세션 동안 채용 공고 문맥이 유지됩니다. 세션을 종료하기 전까지 요구사항은 고정됩니다.")

# ------------------- MAIN UI -------------------
st.title("🧠 HR AI Screening Agent — Streamlit UI")
st.write("**1단계:** 사이드바에서 세션을 생성. **2단계:** 이력서를 평가.")

# ------------------- 탭 구성 -------------------
tab1, tab2, tab3 = st.tabs(["📄 단일 이력서", "📦 일괄 스크리닝", "📊 히스토리 / CSV"])

# ============ TAB 1: 단일 스크리닝 ============
with tab1:
    st.subheader("📄 세션 문맥 기반 이력서 평가")
    if "chat_id" not in st.session_state:
        st.info("먼저 왼쪽에서 **세션 생성/갱신** 버튼을 눌러주세요.")
    else:
        resume_file = st.file_uploader("이력서 (.pdf/.txt)", type=["pdf", "txt"], key="single_resume")
        run = st.button("⚖️ 평가하기")
        if run and resume_file:
            resume_path = save_upload(resume_file, RES_DIR / st.session_state["chat_id"])
            compiled = ensure_graph()
            with st.spinner("평가 중..."):
                try:
                    state = compiled.invoke({
                        "chat_id": st.session_state["chat_id"],
                        "resume_path": resume_path,
                    })
                except Exception as e:
                    st.error(f"에러: {e}")
                    st.stop()

            st.success("✅ 평가 완료!")

            # 결과 표시
            c1, c2, c3 = st.columns([1, 1, 2])
            with c1:
                st.write(decision_badge(state.get("decision", "")))
            with c2:
                score = state.get("score", {}) or {}
                total = score.get("total")
                st.metric("종합 점수", total if total is not None else "—")
            with c3:
                df_score = pretty_score(score)
                if not df_score.empty:
                    st.dataframe(df_score, use_container_width=True, hide_index=True)

            colR, colI = st.columns(2)
            with colR:
                st.markdown("**판단 근거 (reasons):**")
                reasons = state.get("reasons") or state.get("reason") or []
                if isinstance(reasons, str):
                    st.write(reasons)
                else:
                    for r in reasons[:10]:
                        st.markdown(f"- {r}")
            with colI:
                st.markdown("**개선 제안 (improvements):**")
                improvements = state.get("improvements") or []
                if isinstance(improvements, str):
                    st.write(improvements)
                else:
                    for r in improvements[:10]:
                        st.markdown(f"- {r}")

            st.markdown("---")
            st.json({k: v for k, v in state.items() if k not in ["resume_text"]})

# ============ TAB 2: 일괄 스크리닝 ============
with tab2:
    st.subheader("📦 여러 이력서 일괄 평가")
    if "chat_id" not in st.session_state:
        st.info("먼저 세션을 생성하세요.")
    else:
        files = st.file_uploader("이력서들 (.pdf/.txt)", type=["pdf", "txt"], accept_multiple_files=True)
        run_batch = st.button("🚀 일괄 평가")
        if run_batch and files:
            compiled = ensure_graph()
            rows: List[Dict[str, Any]] = []
            with st.spinner("일괄 평가 중..."):
                for f in files:
                    try:
                        rp = save_upload(f, RES_DIR / st.session_state["chat_id"])
                        s = compiled.invoke({"chat_id": st.session_state["chat_id"], "resume_path": rp})
                        score = s.get("score") or {}
                        total = score.get("total") if isinstance(score, dict) else None
                        rows.append({
                            "file": Path(rp).name,
                            "decision": s.get("decision", ""),
                            "total": total,
                            "reasons": json.dumps(s.get("reasons", []), ensure_ascii=False),
                        })
                    except Exception as e:
                        rows.append({"file": f.name, "decision": "ERROR", "total": "", "reasons": str(e)})
            if rows:
                df = pd.DataFrame(rows)
                st.dataframe(df, use_container_width=True, hide_index=True)
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ CSV로 다운로드", csv, "screenings.csv", "text/csv")

# ============ TAB 3: 히스토리 / CSV ============
with tab3:
    st.subheader("📊 히스토리")
    if "chat_id" not in st.session_state:
        st.info("세션을 생성하세요.")
    else:
        sess_dir = RES_DIR / st.session_state["chat_id"]
        if not sess_dir.exists():
            st.write("아직 파일이 없습니다.")
        else:
            files = sorted(sess_dir.glob("*"))
            st.write(f"**{st.session_state['chat_id']}** 세션의 이력서 목록:")
            for f in files:
                st.markdown(f"- `{f.name}`")
    st.caption("CSV 파일은 일괄 스크리닝 탭에서 내려받을 수 있습니다.")
