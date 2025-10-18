from pathlib import Path
from typing import Dict
from langchain_community.document_loaders import PyPDFLoader, TextLoader

def loader_node(state: Dict) -> Dict:
    """
    이력서 로더 노드
    - 업로드된 이력서 파일(.pdf/.txt)을 읽어 텍스트를 반환한다.
    - 다음 단계인 extractor_node가 이 텍스트를 구조화한다.
    
    입력:
        state["resume_path"]: 업로드된 이력서 파일 경로
    출력:
        {"resume_text": str}
    """
    resume_path = state.get("resume_path")
    if not resume_path:
        raise ValueError("resume_path가 제공되지 않았습니다.")

    p = Path(resume_path)
    if not p.exists():
        raise FileNotFoundError(f"이력서 파일을 찾을 수 없습니다: {p}")

    # 파일 확장자에 따라 로더 선택
    if p.suffix.lower() == ".pdf":
        loader = PyPDFLoader(str(p))
        docs = loader.load()
        resume_text = "\n".join([d.page_content for d in docs])
    else:
        loader = TextLoader(str(p), encoding="utf-8")
        docs = loader.load()
        resume_text = "\n".join([d.page_content for d in docs])

    return {"resume_text": resume_text}
