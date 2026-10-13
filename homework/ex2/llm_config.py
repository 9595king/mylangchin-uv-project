# 수업 노트북(001-2_OutPutParser종류.ipynb)의 "LLM 설정" 셀을 .py 파일용으로 옮긴 공통 설정 모듈
# 2-1 ~ 2-4 파일이 모두 `from llm_config import get_llm` 으로 같은 설정을 사용함
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI

# .env 파일을 불러와서 환경 변수로 설정
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 사용할 LLM 제공자 선택: "groq" 또는 "openai" — 이 값만 바꾸면 2-1 ~ 2-4 모든 파일의 모델이 바뀜
PROVIDER = "openai"

# 제공자별 설정 (Groq는 OpenAI 호환 API라서 base_url만 다르고, OpenAI는 base_url 생략 시 기본 엔드포인트 사용)
LLM_CONFIGS = {
    "groq": {
        "api_key": GROQ_API_KEY,
        "base_url": "https://api.groq.com/openai/v1",  # Groq API 엔드포인트
        "model": "openai/gpt-oss-120b",
    },
    "openai": {
        "api_key": OPENAI_API_KEY,
        "base_url": None,
        "model": "gpt-4o-mini",  # 대안: "gpt-4o"
    },
}
MAX_RETRIES = 5  # 503 등 일시 장애 자동 재시도 횟수


def get_llm(temperature: float = 0.5, seed: int | None = None, provider: str | None = None, verbose: bool = True, **kwargs) -> ChatOpenAI:
    """선택한 제공자(기본값: PROVIDER)의 ChatOpenAI 인스턴스를 생성합니다. verbose=True 이면 생성된 모델 정보를 출력합니다.
    kwargs: ChatOpenAI 에 그대로 전달할 추가 옵션 (예: timeout=60)"""
    provider = provider or PROVIDER
    if provider not in LLM_CONFIGS:
        raise ValueError(f"지원하지 않는 제공자: {provider} (사용 가능: {list(LLM_CONFIGS)})")
    config = LLM_CONFIGS[provider]
    if not config["api_key"]:
        raise RuntimeError(f".env 파일에 {provider.upper()}_API_KEY 가 없습니다.")
    llm = ChatOpenAI(
        **config,
        temperature=temperature,
        seed=seed,  # 값을 고정하면 재현성이 향상됨 (완전히 동일한 결과는 보장되지 않음)
        max_retries=MAX_RETRIES,
        **kwargs,
    )
    if verbose:
        # 생성된 모델 정보 출력 (API 키는 출력하지 않음)
        print(f"[LLM 생성] provider={provider} | model={llm.model_name}")
    return llm